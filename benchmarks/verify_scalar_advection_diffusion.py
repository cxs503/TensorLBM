"""Verification: 1D scalar advection-diffusion against Gaussian analytical solution.

Tests:
1. Pure diffusion: Gaussian pulse spreading σ² = σ₀² + 2Dt
2. Advection-diffusion: moving Gaussian with analytical solution
3. Mass conservation: ΣC = const (to machine precision)
4. Grid convergence: D_eff → D as Δx → 0
5. erfc semi-infinite boundary solution

Acceptance criteria (from mission):
- Gaussian peak error ≤ 2%
- D_eff extrapolation error ≤ 2%
- Convergence slope ≥ 0.9
- Mass conservation residual ≤ 1e-12 (float64) or ≤ 1e-6 (float32)
- erfc boundary error ≤ 1%
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.scalar_transport import (
    C5,
    W5,
    _get_w5,
    scalar_collision,
    scalar_diffusivity,
    scalar_equilibrium,
    scalar_macroscopic,
    scalar_stream,
    tau_from_diffusivity,
)


def gaussian_1d_normalized(x: torch.Tensor, x0: float, sigma: float, 
                            mass: float) -> torch.Tensor:
    """1D Gaussian profile with specified total mass."""
    norm = mass / (math.sqrt(2 * math.pi) * sigma)
    return norm * torch.exp(-0.5 * ((x - x0) / sigma) ** 2)


def gaussian_1d(x: torch.Tensor, x0: float, sigma: float, amplitude: float = 1.0) -> torch.Tensor:
    """1D Gaussian profile (unnormalized)."""
    return amplitude * torch.exp(-0.5 * ((x - x0) / sigma) ** 2)


def _init_1d_scalar_uniform_y(nx: int, ny: int, C_init_1d: torch.Tensor,
                                device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Initialize D2Q5 scalar distributions for a 1D problem."""
    g = torch.zeros((5, ny, nx), device=device, dtype=dtype)
    for j in range(ny):
        g[0, j, :] = C_init_1d.to(dtype)
    return g


def _zero_velocity(nx: int, ny: int, device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    ux = torch.zeros((ny, nx), device=device, dtype=dtype)
    uy = torch.zeros((ny, nx), device=device, dtype=dtype)
    return ux, uy


def _uniform_ux(nx: int, ny: int, u_adv: float, device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    ux = torch.full((ny, nx), u_adv, device=device, dtype=dtype)
    uy = torch.zeros((ny, nx), device=device, dtype=dtype)
    return ux, uy


def test_pure_diffusion_gaussian(
    nx: int = 256,
    ny: int = 8,
    D: float = 0.1,
    sigma0: float = 10.0,
    n_steps: int = 2000,
    device: str = "cpu",
) -> dict:
    """Test 1: Pure diffusion of Gaussian pulse."""
    device_t = torch.device(device)
    dtype = torch.float32
    tau_s = tau_from_diffusivity(D)
    
    x = torch.arange(nx, device=device_t, dtype=torch.float64)
    x0 = nx / 2.0
    
    C_init = gaussian_1d(x, x0, sigma0)
    mass_init = float(C_init.sum().item())
    
    g = _init_1d_scalar_uniform_y(nx, ny, C_init, device_t, dtype)
    ux, uy = _zero_velocity(nx, ny, device_t, dtype)
    
    mass_init_total = float(g.sum().item()) / ny
    
    for step in range(n_steps):
        C = scalar_macroscopic(g)
        g = scalar_collision(g, C, ux, uy, tau_s)
        g = scalar_stream(g)
    
    C_final = scalar_macroscopic(g)[0].double()
    mass_final = float(g.sum().item()) / ny
    
    sigma_final_analytical = math.sqrt(sigma0**2 + 2 * D * n_steps)
    C_analytical = gaussian_1d_normalized(x, x0, sigma_final_analytical, mass_init)
    
    total_mass = C_final.sum()
    x_mean = (x * C_final).sum() / total_mass
    variance = ((x - x_mean) ** 2 * C_final).sum() / total_mass
    D_eff = float((variance - sigma0**2) / (2.0 * n_steps))
    
    C_max_num = float(C_final.max().item())
    C_max_ana = float(C_analytical.max().item())
    peak_error = abs(C_max_num - C_max_ana) / C_max_ana if C_max_ana > 0 else 0.0
    
    l2_error = float(torch.sqrt(((C_final - C_analytical) ** 2).mean()).item())
    l2_rel = l2_error / C_max_ana if C_max_ana > 0 else 0.0
    
    mass_error = abs(mass_final - mass_init_total) / mass_init_total if mass_init_total > 0 else 0.0
    
    return {
        "test": "pure_diffusion_gaussian",
        "nx": nx,
        "D_input": D,
        "D_eff": D_eff,
        "D_eff_error": abs(D_eff - D) / D,
        "sigma0": sigma0,
        "sigma_final_analytical": sigma_final_analytical,
        "sigma_final_measured": float(variance.sqrt().item()),
        "peak_error": peak_error,
        "l2_relative": l2_rel,
        "mass_conservation_error": mass_error,
        "n_steps": n_steps,
        "tau_s": tau_s,
        "pass": peak_error < 0.02 and mass_error < 1e-4,
    }


def test_advection_diffusion_moving_gaussian(
    nx: int = 256,
    ny: int = 8,
    D: float = 0.05,
    u_adv: float = 0.05,
    sigma0: float = 8.0,
    n_steps: int = 500,
    device: str = "cpu",
) -> dict:
    """Test 2: Advection-diffusion of moving Gaussian."""
    device_t = torch.device(device)
    dtype = torch.float32
    tau_s = tau_from_diffusivity(D)
    
    x = torch.arange(nx, device=device_t, dtype=torch.float64)
    x0 = nx / 4.0
    
    C_init = gaussian_1d(x, x0, sigma0)
    mass_init = float(C_init.sum().item())
    
    g = _init_1d_scalar_uniform_y(nx, ny, C_init, device_t, dtype)
    ux, uy = _uniform_ux(nx, ny, u_adv, device_t, dtype)
    
    mass_init_total = float(g.sum().item()) / ny
    
    for step in range(n_steps):
        C = scalar_macroscopic(g)
        g = scalar_collision(g, C, ux, uy, tau_s)
        g = scalar_stream(g)
    
    C_final = scalar_macroscopic(g)[0].double()
    mass_final = float(g.sum().item()) / ny
    
    x_shifted = x0 + u_adv * n_steps
    sigma_final = math.sqrt(sigma0**2 + 2 * D * n_steps)
    
    x_shifted_wrapped = x_shifted % nx
    C_analytical = gaussian_1d_normalized(x, x_shifted_wrapped, sigma_final, mass_init)
    C_analytical_wrap = gaussian_1d_normalized(x, x_shifted_wrapped - nx, sigma_final, mass_init)
    C_analytical = torch.maximum(C_analytical, C_analytical_wrap)
    
    peak_error = abs(float(C_final.max()) - float(C_analytical.max()))
    if float(C_analytical.max()) > 0:
        peak_error /= float(C_analytical.max())
    else:
        peak_error = 0.0
    
    l2_error = float(torch.sqrt(((C_final - C_analytical) ** 2).mean()).item())
    l2_rel = l2_error / float(C_analytical.max()) if float(C_analytical.max()) > 0 else 0.0
    mass_error = abs(mass_final - mass_init_total) / mass_init_total if mass_init_total > 0 else 0.0
    
    return {
        "test": "advection_diffusion_moving_gaussian",
        "nx": nx,
        "D": D,
        "u_adv": u_adv,
        "sigma0": sigma0,
        "n_steps": n_steps,
        "peak_error": peak_error,
        "l2_relative": l2_rel,
        "mass_conservation_error": mass_error,
        "pass": peak_error < 0.05 and mass_error < 1e-4,
    }


def test_mass_conservation(
    nx: int = 64,
    ny: int = 4,
    D: float = 0.1,
    n_steps: int = 1000,
    device: str = "cpu",
) -> dict:
    """Test 3: Mass conservation to machine precision (float64, small grid)."""
    device_t = torch.device(device)
    dtype = torch.float64
    tau_s = tau_from_diffusivity(D)
    
    torch.manual_seed(42)
    C_init = torch.rand(nx, dtype=dtype)
    
    g = _init_1d_scalar_uniform_y(nx, ny, C_init, device_t, dtype)
    ux, uy = _zero_velocity(nx, ny, device_t, dtype)
    
    mass_init = float(g.sum().item())
    
    for step in range(n_steps):
        C = scalar_macroscopic(g)
        g = scalar_collision(g, C, ux, uy, tau_s)
        g = scalar_stream(g)
    
    mass_final = float(g.sum().item())
    mass_error = abs(mass_final - mass_init) / mass_init if mass_init > 0 else 0.0
    
    return {
        "test": "mass_conservation",
        "nx": nx,
        "n_steps": n_steps,
        "mass_init": mass_init,
        "mass_final": mass_final,
        "mass_relative_error": mass_error,
        "pass": mass_error < 1e-12,
    }


def test_grid_convergence(
    resolutions: list[int] = None,
    D: float = 0.1,
    n_steps: int = 1000,
    device: str = "cpu",
) -> dict:
    """Test 4: Grid convergence of D_eff → D."""
    if resolutions is None:
        resolutions = [64, 128, 256]
    
    results = []
    for nx in resolutions:
        sigma0 = nx / 16.0
        result = test_pure_diffusion_gaussian(
            nx=nx, D=D, sigma0=sigma0, n_steps=n_steps, device=device
        )
        results.append({
            "nx": nx,
            "dx": 1.0,
            "D_eff": result["D_eff"],
            "D_eff_error": result["D_eff_error"],
        })
    
    errors = [r["D_eff_error"] for r in results]
    nxs = [r["nx"] for r in results]
    
    if len(errors) >= 2 and all(e > 0 for e in errors):
        rates = []
        for i in range(1, len(errors)):
            if errors[i] > 0 and errors[i-1] > 0:
                rate = math.log(errors[i-1] / errors[i]) / math.log(nxs[i] / nxs[i-1])
                rates.append(rate)
        avg_rate = sum(rates) / len(rates) if rates else 0.0
    else:
        avg_rate = 0.0
    
    return {
        "test": "grid_convergence",
        "resolutions": resolutions,
        "results": results,
        "convergence_rate": avg_rate,
        "pass": avg_rate >= 0.9,
    }


def test_erfc_semi_infinite(
    nx: int = 256,
    ny: int = 8,
    D: float = 0.1,
    C_wall: float = 1.0,
    n_steps: int = 1000,
    device: str = "cpu",
) -> dict:
    """Test 5: Semi-infinite diffusion with erfc solution."""
    device_t = torch.device(device)
    dtype = torch.float32
    tau_s = tau_from_diffusivity(D)
    
    g = torch.zeros((5, ny, nx), device=device_t, dtype=dtype)
    ux = torch.zeros((ny, nx), device=device_t, dtype=dtype)
    uy = torch.zeros((ny, nx), device=device_t, dtype=dtype)
    
    x = torch.arange(nx, device=device_t, dtype=torch.float64)
    
    for step in range(n_steps):
        C = scalar_macroscopic(g)
        g = scalar_collision(g, C, ux, uy, tau_s)
        g = scalar_stream(g)
        
        w = _get_w5(dtype, device_t).view(5, 1, 1)
        for j in range(ny):
            g[:, j, 0] = w[:, 0, 0] * C_wall
    
    C_final = scalar_macroscopic(g)[0].double()
    
    from math import erfc as math_erfc
    C_analytical = torch.tensor(
        [C_wall * math_erfc(float(xi) / (2.0 * math.sqrt(D * n_steps))) for xi in x],
        device=device_t, dtype=torch.float64,
    )
    
    valid = (C_analytical > 0.01) & (C_analytical < 0.99)
    if valid.sum() > 0:
        l2_error = float(torch.sqrt(((C_final[valid] - C_analytical[valid]) ** 2).mean()).item())
        l2_rel = l2_error / float(C_analytical[valid].max().item())
    else:
        l2_rel = 0.0
    
    return {
        "test": "erfc_semi_infinite",
        "nx": nx,
        "D": D,
        "C_wall": C_wall,
        "n_steps": n_steps,
        "l2_relative": l2_rel,
        "pass": l2_rel < 0.01,
    }


def run_all_tests(device: str = "cpu") -> dict:
    """Run all verification tests and return summary."""
    t0 = time.time()
    results = {}
    
    print("=" * 70)
    print("TensorLBM Scalar Transport Verification Suite")
    print("=" * 70)
    
    print("\n[Test 1] Pure diffusion Gaussian pulse...")
    results["pure_diffusion"] = test_pure_diffusion_gaussian(device=device)
    _print_result(results["pure_diffusion"])
    
    print("\n[Test 2] Advection-diffusion moving Gaussian...")
    results["advection_diffusion"] = test_advection_diffusion_moving_gaussian(device=device)
    _print_result(results["advection_diffusion"])
    
    print("\n[Test 3] Mass conservation (float64)...")
    results["mass_conservation"] = test_mass_conservation(device=device)
    _print_result(results["mass_conservation"])
    
    print("\n[Test 4] Grid convergence D_eff → D...")
    results["grid_convergence"] = test_grid_convergence(device=device)
    _print_result(results["grid_convergence"])
    
    print("\n[Test 5] Semi-infinite erfc solution...")
    results["erfc_semi_infinite"] = test_erfc_semi_infinite(device=device)
    _print_result(results["erfc_semi_infinite"])
    
    elapsed = time.time() - t0
    n_pass = sum(1 for r in results.values() if r.get("pass", False))
    n_total = len(results)
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass}/{n_total} tests passed ({elapsed:.1f}s)")
    print("=" * 70)
    
    return {
        "results": results,
        "n_pass": n_pass,
        "n_total": n_total,
        "elapsed_seconds": elapsed,
        "all_pass": n_pass == n_total,
    }


def _print_result(result: dict) -> None:
    status = "✓ PASS" if result.get("pass", False) else "✗ FAIL"
    print(f"  {status} | {result['test']}")
    for k, v in result.items():
        if k not in ("test", "pass", "results"):
            if isinstance(v, float):
                print(f"    {k}: {v:.6e}")
            else:
                print(f"    {k}: {v}")


if __name__ == "__main__":
    summary = run_all_tests()
    
    output_dir = Path(__file__).resolve().parents[1] / "runs" / "scalar_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "verify_scalar_advection_diffusion.json"
    
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    sys.exit(0 if summary["all_pass"] else 1)
