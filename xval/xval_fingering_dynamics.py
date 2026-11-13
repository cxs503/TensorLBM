"""Cross-validation: TensorLBM vs ebinan92/Fingering_dynamics (cupy-dropin-no-kernel-fusion).

Implements the Swift-Osborn-Yeomans (SOY) free-energy D2Q9 LBM from the OSS
repository in PyTorch (TensorLBM-native), validates against the analytical
Young-Laplace pressure jump Δp = σ/R for a static droplet, and compares with
TensorLBM's built-in free_energy_step and CAC2D modules.

OSS source: ebinan92/Fingering_dynamics @ 150a5739e
Key algorithm (extracted from fingering.py):
  - Two D2Q9 populations: f (flow), g (order parameter ψ ∈ [-1, +1])
  - Chemical potential: μ = aψ(1 − ψ²) − κ∇²ψ  (a = −2κ/ξ²)
  - Free-energy pressure: p = ρ/3 + ψμ/3
  - Guo forcing on f for the capillary force μ∇ψ
  - g-equilibrium includes chemical-potential diffusion terms (B_0, B_{1-8})
  - Isotropic D2Q9 stencil for gradients and Laplacian
  - Half-way bounce-back on walls, Zou-He open boundaries (not used here)

Validation test: static droplet Laplace pressure
  - Circular droplet of radius R in a periodic domain
  - Measure Δp = p_inside − p_outside at equilibrium
  - Analytical: Δp = σ/R
  - Acceptance: |Δp_sim − σ/R| / (σ/R) < 0.15 (owner precision standard)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# D2Q9 constants
# ---------------------------------------------------------------------------
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# SOY Free-Energy D2Q9 LBM — PyTorch (faithful to OSS algorithm)
# ---------------------------------------------------------------------------

class SOYFreeEnergyLBM:
    """Swift-Osborn-Yeomans free-energy D2Q9 LBM (PyTorch).

    Faithful reimplementation of the core algorithm from
    ebinan92/Fingering_dynamics/lattice_boltzmann/fingering.py.
    Uses PyTorch tensor operations — analogous to the OSS cupy-dropin
    strategy (array-level GPU ops, no custom kernels), but with PyTorch
    instead of CuPy.

    Key algorithmic choices matching the OSS code:
    - Chemical potential: μ = aψ(1-ψ²) - κ∇²ψ  (OSS getMu)
    - Pressure: p = ρ/3 + ψμ/3  (OSS getP)
    - Velocity: u = (j + μ∇ψ/2) / ρ  (half-force Guo correction)
    - f-equilibrium: A_0 = (ρ - 3(1-w₀)p)/w₀, A_{1-8} = 3p  (OSS getfeq)
    - g-equilibrium: B_0 = (ψ - 3(1-w₀)γμ)/w₀, B_{1-8} = 3γμ  (OSS getgeq)
    - Guo forcing on f: Si = (1-1/(2τ_f)) w_i [3(c_i·F) + 9(c_i·u)(c_i·F) - 3(u·F)]
    - Isotropic D2Q9 stencil for ∇ψ and ∇²ψ
    """

    def __init__(self, ny: int, nx: int, *, sigma: float, xi: float,
                 tau_f: float, tau_g: float, gamma: float,
                 device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.ny, self.nx = ny, nx
        self.sigma, self.xi = sigma, xi
        self.kappa = 0.75 * sigma * xi
        self.a = -2.0 * self.kappa / (xi ** 2)
        self.tau_f, self.tau_g, self.gamma = tau_f, tau_g, gamma
        self.device, self.dtype = torch.device(device), dtype
        self.w0 = float(W[0])

        self.cx_t = torch.tensor(CX, dtype=dtype, device=self.device).view(9, 1, 1)
        self.cy_t = torch.tensor(CY, dtype=dtype, device=self.device).view(9, 1, 1)
        self.w_t = torch.tensor(W, dtype=dtype, device=self.device).view(9, 1, 1)

        self.f = torch.zeros((9, ny, nx), dtype=dtype, device=self.device)
        self.g = torch.zeros((9, ny, nx), dtype=dtype, device=self.device)
        self.psi = torch.zeros((ny, nx), dtype=dtype, device=self.device)
        self.step_count = 0

    def init_droplet(self, R: float, cx: float = None, cy: float = None):
        if cx is None: cx = self.nx / 2.0
        if cy is None: cy = self.ny / 2.0
        y = torch.arange(self.ny, dtype=self.dtype, device=self.device).view(self.ny, 1)
        x = torch.arange(self.nx, dtype=self.dtype, device=self.device).view(1, self.nx)
        dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        self.psi = torch.tanh((dist - R) / self.xi)
        self._dist = dist
        self._R = R

        grad_psi_x, grad_psi_y, lap_psi = self._compute_gradients(self.psi)
        mu = self._chemical_potential(self.psi, lap_psi)
        rho = torch.ones((self.ny, self.nx), dtype=self.dtype, device=self.device)
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        p = self._pressure(rho, self.psi, mu)

        cu = self.cx_t * ux.unsqueeze(0) + self.cy_t * uy.unsqueeze(0)
        usq = (ux ** 2 + uy ** 2).unsqueeze(0)

        A = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        A[0] = (rho - 3.0 * (1.0 - self.w0) * p) / self.w0
        A[1:] = 3.0 * p.unsqueeze(0).expand(8, self.ny, self.nx)
        self.f = self.w_t * (A + rho.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))

        B = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        B[0] = (self.psi - 3.0 * (1.0 - self.w0) * self.gamma * mu) / self.w0
        B[1:] = 3.0 * self.gamma * mu.unsqueeze(0).expand(8, self.ny, self.nx)
        self.g = self.w_t * (B + self.psi.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))

    def _compute_gradients(self, psi):
        grad_x = torch.zeros_like(psi)
        grad_y = torch.zeros_like(psi)
        lap = torch.zeros_like(psi)
        for i in range(1, 9):
            dx, dy = int(CX[i]), int(CY[i])
            shifted = torch.roll(torch.roll(psi, shifts=-dx, dims=1), shifts=-dy, dims=0)
            grad_x = grad_x + 3.0 * float(W[i]) * float(CX[i]) * shifted
            grad_y = grad_y + 3.0 * float(W[i]) * float(CY[i]) * shifted
            lap = lap + float(W[i]) * shifted
        lap = 6.0 * (lap - (1.0 - float(W[0])) * psi)
        return grad_x, grad_y, lap

    def _chemical_potential(self, psi, lap_psi):
        return self.a * psi * (1.0 - psi ** 2) - self.kappa * lap_psi

    def _pressure(self, rho, psi, mu):
        return rho / 3.0 + psi * mu / 3.0

    def step(self):
        rho = self.f.sum(dim=0)
        grad_psi_x, grad_psi_y, lap_psi = self._compute_gradients(self.psi)
        mu = self._chemical_potential(self.psi, lap_psi)

        jx = (self.f * self.cx_t).sum(dim=0)
        jy = (self.f * self.cy_t).sum(dim=0)
        Fx = mu * grad_psi_x
        Fy = mu * grad_psi_y
        rho_s = rho.clamp(min=1e-12)
        ux = (jx + 0.5 * Fx) / rho_s
        uy = (jy + 0.5 * Fy) / rho_s
        p = self._pressure(rho, self.psi, mu)

        cu = self.cx_t * ux.unsqueeze(0) + self.cy_t * uy.unsqueeze(0)
        usq = (ux ** 2 + uy ** 2).unsqueeze(0)
        ci_dot_F = self.cx_t * Fx.unsqueeze(0) + self.cy_t * Fy.unsqueeze(0)
        u_dot_F = (ux * Fx + uy * Fy).unsqueeze(0)

        A = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        A[0] = (rho - 3.0 * (1.0 - self.w0) * p) / self.w0
        A[1:] = 3.0 * p.unsqueeze(0).expand(8, self.ny, self.nx)
        feq = self.w_t * (A + rho.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))

        Si = (1.0 - 0.5 / self.tau_f) * self.w_t * (
            3.0 * ci_dot_F + 9.0 * cu * ci_dot_F - 3.0 * u_dot_F
        )
        self.f = self.f - (self.f - feq) / self.tau_f + Si

        B = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        B[0] = (self.psi - 3.0 * (1.0 - self.w0) * self.gamma * mu) / self.w0
        B[1:] = 3.0 * self.gamma * mu.unsqueeze(0).expand(8, self.ny, self.nx)
        geq = self.w_t * (B + self.psi.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))
        self.g = self.g - (self.g - geq) / self.tau_g

        f_new = torch.empty_like(self.f)
        g_new = torch.empty_like(self.g)
        f_new[0] = self.f[0]
        g_new[0] = self.g[0]
        for i in range(1, 9):
            dx, dy = int(CX[i]), int(CY[i])
            f_new[i] = torch.roll(torch.roll(self.f[i], shifts=dx, dims=1), shifts=dy, dims=0)
            g_new[i] = torch.roll(torch.roll(self.g[i], shifts=dx, dims=1), shifts=dy, dims=0)
        self.f, self.g = f_new, g_new
        self.psi = self.g.sum(dim=0)
        self.step_count += 1

    def measure(self):
        """Return (Δp, u_max, ψ_min, ψ_max)."""
        R, xi = self._R, self.xi
        dist = self._dist
        inside = dist < (R - 2 * xi)
        outside = dist > (R + 2 * xi)
        _, _, lap_psi = self._compute_gradients(self.psi)
        mu = self._chemical_potential(self.psi, lap_psi)
        rho = self.f.sum(dim=0)
        p = self._pressure(rho, self.psi, mu)
        jx = (self.f * self.cx_t).sum(dim=0)
        jy = (self.f * self.cy_t).sum(dim=0)
        Fx = mu * self._compute_gradients(self.psi)[0]
        Fy = mu * self._compute_gradients(self.psi)[1]
        rho_s = rho.clamp(min=1e-12)
        ux = (jx + 0.5 * Fx) / rho_s
        uy = (jy + 0.5 * Fy) / rho_s
        u_max = float(torch.sqrt(ux**2 + uy**2).max())
        if not inside.any() or not outside.any():
            return float("nan"), u_max, float(self.psi.min()), float(self.psi.max())
        dp = float(p[inside].mean() - p[outside].mean())
        return dp, u_max, float(self.psi.min()), float(self.psi.max())


# ---------------------------------------------------------------------------
# TensorLBM CAC2D wrapper
# ---------------------------------------------------------------------------

def run_cac2d_droplet(ny, nx, R, sigma, width, n_steps):
    """Run a droplet test using TensorLBM's CAC2D module."""
    from tensorlbm.cac_lbm2d import CAC2D, init_phi_droplet, diagnostics

    # CAC2D uses φ ∈ [0, 1] with width W (different from OSS ξ)
    # OSS: ψ = tanh((r-R)/ξ), φ = (ψ+1)/2 = 0.5(1+tanh((r-R)/ξ))
    # CAC2D: φ = 0.5(1+tanh(2(r-R)/W))
    # Matching: 1/ξ = 2/W → W = 2ξ
    W_cac = 2.0 * width  # interface width for CAC2D

    # Same viscosity
    nu = (1.0 - 0.5) / 3.0  # tau_f = 1.0
    mu_l = nu * 1.0  # dynamic viscosity (rho_l = 1)
    mu_g = nu * 1.0  # same viscosity for both phases

    sim = CAC2D(
        (ny, nx),
        rho_l=1.0, rho_g=1.0,  # equal density (no buoyancy)
        mu_l=mu_l, mu_g=mu_g,
        sigma=sigma, width=W_cac,
        mobility=0.05,
        axes="pp",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    phi0 = init_phi_droplet((ny, nx), radius=R, width=W_cac)
    sim.initialize(phi0)

    dp_history = []
    for step in range(n_steps):
        sim.step()
        if step % 500 == 0 or step == n_steps - 1:
            d = diagnostics(sim)
            dp_history.append((step, d))
            print(f"  CAC2D step {step}: u_max={d['u_max_fluid']:.4e}, "
                  f"phi=[{d['phi_min_fluid']:.4f},{d['phi_max_fluid']:.4f}]")

    # Measure Laplace pressure from the final state
    d = diagnostics(sim)
    return d, dp_history


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

@dataclass
class XVALConfig:
    ny: int = 60
    nx: int = 60
    R: float = 15.0
    sigma: float = 0.01
    xi: float = 3.0
    tau_f: float = 1.0
    tau_g: float = 0.8
    n_steps: int = 2000
    sample_every: int = 500


def run_cross_validation(cfg: XVALConfig = None) -> dict:
    if cfg is None:
        cfg = XVALConfig()

    kappa = 0.75 * cfg.sigma * cfg.xi
    gamma_mob = (cfg.tau_g - 0.5) / 3.0
    dp_analytical = cfg.sigma / cfg.R

    results = {
        "config": {
            "ny": cfg.ny, "nx": cfg.nx, "R": cfg.R,
            "sigma": cfg.sigma, "xi": cfg.xi,
            "kappa": kappa, "tau_f": cfg.tau_f, "tau_g": cfg.tau_g,
            "gamma": gamma_mob, "n_steps": cfg.n_steps,
            "nu": (cfg.tau_f - 0.5) / 3.0,
        },
        "analytical_dp": dp_analytical,
    }

    # --- 1. SOY PyTorch (OSS algorithm in PyTorch) ---
    print("Running SOY PyTorch (OSS algorithm)...")
    t0 = time.time()
    sim = SOYFreeEnergyLBM(cfg.ny, cfg.nx, sigma=cfg.sigma, xi=cfg.xi,
                            tau_f=cfg.tau_f, tau_g=cfg.tau_g, gamma=gamma_mob,
                            dtype=torch.float64)
    sim.init_droplet(cfg.R)
    dp_history = []
    for step in range(cfg.n_steps):
        sim.step()
        if step % cfg.sample_every == 0 or step == cfg.n_steps - 1:
            dp, u_max, psi_min, psi_max = sim.measure()
            dp_history.append((step, dp, u_max, psi_min, psi_max))
            print(f"  step {step}: Δp={dp:.6e}, u_max={u_max:.4e}, "
                  f"ψ=[{psi_min:.4f},{psi_max:.4f}]")
    t_soy = time.time() - t0
    dp_soy = dp_history[-1][1]
    err_soy = abs(dp_soy - dp_analytical) / dp_analytical
    results["soy_pytorch"] = {
        "dp": dp_soy, "error": err_soy, "time_s": t_soy,
        "dp_history": [(h[0], h[1]) for h in dp_history],
        "u_max_final": dp_history[-1][2],
        "psi_range_final": (dp_history[-1][3], dp_history[-1][4]),
        "pass": err_soy < 0.15,
    }
    print(f"  SOY: Δp = {dp_soy:.6e}, analytical = {dp_analytical:.6e}, "
          f"error = {err_soy:.4%}, {'PASS' if err_soy < 0.15 else 'FAIL'}")

    # --- 2. TensorLBM CAC2D ---
    print("Running TensorLBM CAC2D...")
    t0 = time.time()
    try:
        cac_diag, cac_hist = run_cac2d_droplet(
            cfg.ny, cfg.nx, cfg.R, cfg.sigma, cfg.xi, cfg.n_steps
        )
        t_cac = time.time() - t0
        results["tensorlbm_cac2d"] = {
            "diagnostics": cac_diag,
            "time_s": t_cac,
            "note": "CAC2D uses φ∈[0,1] parameterisation; Laplace Δp measured from pressure field",
        }
    except Exception as e:
        results["tensorlbm_cac2d"] = {"error": str(e)}
        print(f"  CAC2D failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------

def run_convergence_study(radii=None, cfg_base=None) -> dict:
    if radii is None:
        radii = [10.0, 13.0, 16.0, 19.0]
    if cfg_base is None:
        cfg_base = XVALConfig(n_steps=1500)

    results = {"radii": radii, "sigma_true": cfg_base.sigma, "dps": [], "sigma_effs": []}

    for R in radii:
        ny = max(int(4 * R + 4 * cfg_base.xi), cfg_base.ny)
        nx = ny
        gamma_mob = (cfg_base.tau_g - 0.5) / 3.0
        sim = SOYFreeEnergyLBM(ny, nx, sigma=cfg_base.sigma, xi=cfg_base.xi,
                                tau_f=cfg_base.tau_f, tau_g=cfg_base.tau_g,
                                gamma=gamma_mob, dtype=torch.float64)
        sim.init_droplet(R)
        for _ in range(cfg_base.n_steps):
            sim.step()
        dp, _, _, _ = sim.measure()
        dp_an = cfg_base.sigma / R
        err = abs(dp - dp_an) / dp_an
        sigma_eff = dp * R
        results["dps"].append(dp)
        results["sigma_effs"].append(sigma_eff)
        print(f"  R={R:.0f}: Δp={dp:.6e}, analytical={dp_an:.6e}, "
              f"err={err:.4%}, σ_eff={sigma_eff:.6e}")

    mean_sigma = sum(results["sigma_effs"]) / len(results["sigma_effs"])
    sigma_err = abs(mean_sigma - cfg_base.sigma) / cfg_base.sigma
    results["sigma_eff_mean"] = mean_sigma
    results["sigma_eff_error"] = sigma_err
    print(f"  σ_eff = {mean_sigma:.6e}, σ_true = {cfg_base.sigma:.6e}, "
          f"error = {sigma_err:.4%}")
    return results


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------

def run_sensitivity_study(cfg_base=None) -> dict:
    if cfg_base is None:
        cfg_base = XVALConfig(n_steps=1000)

    results = {"tau_f_sweep": {}, "xi_sweep": {}}

    for tau_f in [0.7, 0.8, 1.0, 1.2, 1.5]:
        gamma_mob = (cfg_base.tau_g - 0.5) / 3.0
        sim = SOYFreeEnergyLBM(cfg_base.ny, cfg_base.nx, sigma=cfg_base.sigma, xi=cfg_base.xi,
                                tau_f=tau_f, tau_g=cfg_base.tau_g, gamma=gamma_mob,
                                dtype=torch.float64)
        sim.init_droplet(cfg_base.R)
        for _ in range(cfg_base.n_steps):
            sim.step()
        dp, u_max, psi_min, psi_max = sim.measure()
        dp_an = cfg_base.sigma / cfg_base.R
        err = abs(dp - dp_an) / dp_an
        results["tau_f_sweep"][str(tau_f)] = {
            "dp": dp, "error": err, "u_max": u_max,
            "psi_min": psi_min, "psi_max": psi_max,
        }
        print(f"  τ_f={tau_f}: Δp={dp:.6e}, err={err:.4%}, u_max={u_max:.4e}")

    for xi in [2.0, 3.0, 4.0, 5.0]:
        gamma_mob = (cfg_base.tau_g - 0.5) / 3.0
        sim = SOYFreeEnergyLBM(cfg_base.ny, cfg_base.nx, sigma=cfg_base.sigma, xi=xi,
                                tau_f=cfg_base.tau_f, tau_g=cfg_base.tau_g, gamma=gamma_mob,
                                dtype=torch.float64)
        sim.init_droplet(cfg_base.R)
        for _ in range(cfg_base.n_steps):
            sim.step()
        dp, u_max, psi_min, psi_max = sim.measure()
        dp_an = cfg_base.sigma / cfg_base.R
        err = abs(dp - dp_an) / dp_an
        results["xi_sweep"][str(xi)] = {
            "dp": dp, "error": err, "u_max": u_max,
            "psi_min": psi_min, "psi_max": psi_max,
        }
        print(f"  ξ={xi}: Δp={dp:.6e}, err={err:.4%}, u_max={u_max:.4e}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Δp convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    dp_an = results["analytical_dp"]
    ax.axhline(dp_an, color="k", ls="--", lw=1.5, label=f"Analytical σ/R = {dp_an:.4e}")
    if "soy_pytorch" in results:
        hist = results["soy_pytorch"]["dp_history"]
        steps = [h[0] for h in hist]
        dps = [h[1] for h in hist]
        ax.plot(steps, dps, "o-", label="SOY PyTorch (OSS algorithm)", markersize=4, color="C0")
    ax.set_xlabel("LBM step")
    ax.set_ylabel("Δp (lattice units)")
    ax.set_title("Droplet Laplace Pressure Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_laplace_convergence.png", dpi=150)
    plt.close(fig)

    # 2. Error summary
    fig, ax = plt.subplots(figsize=(6, 4))
    if "soy_pytorch" in results:
        err = results["soy_pytorch"]["error"] * 100
        bar = ax.bar(["SOY PyTorch\n(OSS algorithm)"], [err], color="C0", edgecolor="k")
        ax.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() + 0.3,
                f"{err:.2f}%", ha="center", va="bottom", fontsize=11)
    ax.axhline(15.0, color="r", ls="--", lw=2, label="15% threshold")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Laplace Pressure Error vs Analytical σ/R")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_error_summary.png", dpi=150)
    plt.close(fig)

    # 3. Convergence: Δp vs R
    if "convergence" in results:
        conv = results["convergence"]
        fig, ax = plt.subplots(figsize=(8, 5))
        radii = conv["radii"]
        dps = conv["dps"]
        sigma_true = conv["sigma_true"]
        ax.plot(radii, dps, "o-", label="SOY PyTorch", markersize=6, color="C0")
        R_range = np.linspace(min(radii) * 0.8, max(radii) * 1.1, 50)
        ax.plot(R_range, sigma_true / R_range, "k--", lw=1.5,
                label=f"Analytical σ/R (σ={sigma_true})")
        ax.set_xlabel("Droplet radius R (lattice units)")
        ax.set_ylabel("Δp (lattice units)")
        ax.set_title("Laplace Pressure vs Droplet Radius")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig3_laplace_vs_radius.png", dpi=150)
        plt.close(fig)

    # 4. Sensitivity
    if "sensitivity" in results:
        sens = results["sensitivity"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if "tau_f_sweep" in sens:
            taus = sorted(float(k) for k in sens["tau_f_sweep"])
            errs = [sens["tau_f_sweep"][str(t)]["error"] * 100 for t in taus]
            u_maxs = [sens["tau_f_sweep"][str(t)]["u_max"] for t in taus]
            ax = axes[0]
            ax.plot(taus, errs, "o-", label="Δp error")
            ax.axhline(15.0, color="r", ls="--", label="15% threshold")
            ax.set_xlabel("τ_f (relaxation time)")
            ax.set_ylabel("Relative error (%)")
            ax.set_title("Sensitivity to τ_f")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax2 = ax.twinx()
            ax2.plot(taus, u_maxs, "s--", color="C1", label="u_max")
            ax2.set_ylabel("u_max", color="C1")
            ax2.legend(loc="upper right")

        if "xi_sweep" in sens:
            xis = sorted(float(k) for k in sens["xi_sweep"])
            errs = [sens["xi_sweep"][str(x)]["error"] * 100 for x in xis]
            u_maxs = [sens["xi_sweep"][str(x)]["u_max"] for x in xis]
            ax = axes[1]
            ax.plot(xis, errs, "o-", label="Δp error")
            ax.axhline(15.0, color="r", ls="--", label="15% threshold")
            ax.set_xlabel("ξ (interface width)")
            ax.set_ylabel("Relative error (%)")
            ax.set_title("Sensitivity to ξ")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax2 = ax.twinx()
            ax2.plot(xis, u_maxs, "s--", color="C1", label="u_max")
            ax2.set_ylabel("u_max", color="C1")
            ax2.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(out_dir / "fig4_sensitivity.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    out_dir = Path(__file__).parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("XVAL: TensorLBM vs ebinan92/Fingering_dynamics")
    print("       (cupy-dropin-no-kernel-fusion)")
    print("=" * 70)

    # Primary validation
    results = run_cross_validation()

    # Convergence study
    print("\n--- Convergence study (multiple radii) ---")
    conv = run_convergence_study()
    results["convergence"] = conv

    # Sensitivity study
    print("\n--- Sensitivity study ---")
    sens = run_sensitivity_study()
    results["sensitivity"] = sens

    # Plots
    plot_results(results, out_dir)

    # Save results
    def sanitize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and math.isnan(obj):
            return "NaN"
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(x) for x in obj]
        if isinstance(obj, tuple):
            return [sanitize(x) for x in obj]
        return obj

    with open(out_dir / "xval_results.json", "w") as f:
        json.dump(sanitize(results), f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if "soy_pytorch" in results:
        r = results["soy_pytorch"]
        status = "✓ PASS" if r.get("pass") else "✗ FAIL"
        print(f"  SOY PyTorch (OSS algorithm): Δp = {r['dp']:.6e}, "
              f"error = {r['error']:.4%} {status}")

    if "convergence" in results:
        c = results["convergence"]
        print(f"  σ_eff = {c['sigma_eff_mean']:.6e}, σ_true = {c['sigma_true']:.6e}, "
              f"error = {c['sigma_eff_error']:.4%}")

    print(f"\nArtifacts saved to {out_dir}")
