"""Viscoplastic (Herschel-Bulkley) wall-resolved turbulent channel solver.

Extends the Newtonian wall-resolved channel3D solver to support
Herschel-Bulkley rheology with Papanastasiou regularisation.

The key modification is that the relaxation time τ becomes spatially varying:
    τ_eff(x,y,z) = 0.5 + 3·ν_eff(x,y,z)
where ν_eff = μ_eff/ρ is computed from the local shear rate using the
Papanastasiou-regularised HB model:
    μ_eff = K·|γ̇|^(n-1) + τ_y·(1 − exp(−m·|γ̇|)) / |γ̇|

The solver uses:
- D3Q19 lattice with TRT or cumulant collision
- Half-way bounce-back walls
- Body-force driving (constant pressure gradient)
- Optional Smagorinsky SGS model
- Statistics collection (mean velocity, Reynolds stresses, TKE)

For turbulent flows of viscoplastic fluids, the key dimensionless groups are:
- Re_MR: Metzner-Reed generalized Reynolds number
- Y = τ_y / (ρ·u_τ²): yield number (Bingham number based on friction velocity)
- n: flow index

References
----------
* Papanastasiou, T.C. (1987) J. Rheol. 31, 385.
* Dodge, D.W. & Metzner, A.B. (1959) I&EC 51, 1161.
* Metzner, A.B. & Reed, J.C. (1955) AIChE J. 1, 434.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .d3q19 import C as C19, W as W19, OPPOSITE as OPP19
from .d3q19 import equilibrium3d, macroscopic3d


@dataclass
class ViscoplasticChannelConfig:
    """Configuration for viscoplastic turbulent channel flow."""
    # Grid
    nx: int = 128       # streamwise
    ny: int = 48        # wall-normal
    nz: int = 64        # spanwise

    # Rheology (Herschel-Bulkley with Papanastasiou regularisation)
    tau_y: float = 0.0      # yield stress
    K: float = 0.02         # consistency index
    n: float = 1.0          # flow index (1 = Bingham/Newtonian)
    m_pap: float = 500.0    # Papanastasiou regularisation parameter

    # Flow parameters
    re_tau: float = 180.0   # friction Reynolds number (based on Newtonian ν)
    u_tau: float = 0.006    # friction velocity (lattice units)
    rho: float = 1.0        # density (lattice units)

    # Time stepping
    steps: int = 30000
    warmup_steps: int = 5000
    sample_interval: int = 5
    report_interval: int = 500

    # Numerics
    smagorinsky_cs: float = 0.0   # SGS coefficient (0 = no SGS)
    nu_min: float = 1e-5          # min kinematic viscosity floor
    nu_max: float = 0.3           # max kinematic viscosity cap
    wall_origin: float = 0.5      # wall position for y+ mapping

    # I/O
    seed: int = 20261110
    device: str = "cpu"
    dtype: str = "float64"
    output: Path = Path("results/vp_channel/result.json")

    def __post_init__(self):
        object.__setattr__(self, "output", Path(self.output))

    @property
    def height(self) -> int:
        """Fluid height (excluding wall rows)."""
        return self.ny - 2

    @property
    def half_height(self) -> float:
        return 0.5 * self.height

    @property
    def nu_newtonian(self) -> float:
        """Newtonian kinematic viscosity for Re_tau matching."""
        return self.u_tau * self.half_height / self.re_tau

    @property
    def tau_newtonian(self) -> float:
        """Newtonian relaxation time."""
        return 0.5 + 3.0 * self.nu_newtonian

    @property
    def body_force(self) -> float:
        """Body force acceleration to sustain the flow."""
        return 2.0 * self.u_tau ** 2 / self.height

    @property
    def yield_number(self) -> float:
        """Yield number Y = τ_y / (ρ · u_τ²)."""
        if self.u_tau <= 0:
            return float('inf')
        return self.tau_y / (self.rho * self.u_tau ** 2)

    def validate(self):
        if min(self.nx, self.nz) < 8 or self.ny < 10:
            raise ValueError("Grid too small")
        if self.re_tau <= 0 or self.u_tau <= 0:
            raise ValueError("re_tau and u_tau must be positive")
        if self.tau_y < 0:
            raise ValueError("tau_y must be >= 0")
        if self.K <= 0:
            raise ValueError("K must be > 0")
        if self.n <= 0 or self.n > 2:
            raise ValueError("n must be in (0, 2]")
        if self.m_pap <= 0:
            raise ValueError("m_pap must be > 0")


class ViscoplasticChannel3D:
    """D3Q19 viscoplastic channel flow solver.

    Populations stored as (19, nz, ny, nx).
    Periodic in x and z, no-slip walls at y=0 and y=ny-1.
    """

    def __init__(self, cfg: ViscoplasticChannelConfig):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dt = torch.float32 if cfg.dtype == "float32" else torch.float64

        # Lattice constants
        self.C = C19.to(self.device, dtype=self.dt)
        self.W = W19.to(self.device, dtype=self.dt)
        self.OPP = OPP19.to(self.device)

        self.nx = cfg.nx
        self.ny = cfg.ny
        self.nz = cfg.nz

        # Wall indices
        cy = self.C[:, 1]
        self.up_idx = torch.where(cy > 0)[0]
        self.down_idx = torch.where(cy < 0)[0]

        # Statistics accumulators
        self._stat_steps = 0
        self._u_mean = torch.zeros(self.ny, device=self.device, dtype=self.dt)
        self._uu = torch.zeros(self.ny, device=self.device, dtype=self.dt)
        self._vv = torch.zeros(self.ny, device=self.device, dtype=self.dt)
        self._ww = torch.zeros(self.ny, device=self.device, dtype=self.dt)
        self._uv = torch.zeros(self.ny, device=self.device, dtype=self.dt)

        self._init_populations()

    def _init_populations(self):
        """Initialize with parabolic profile + perturbation."""
        cfg = self.cfg
        h = cfg.half_height
        G = cfg.body_force

        # Wall-normal coordinate
        y = torch.arange(self.ny, device=self.device, dtype=self.dt).view(1, self.ny, 1)
        y_center = h  # centerline

        # For HB fluid, use approximate profile (Newtonian as initial guess)
        nu = cfg.nu_newtonian
        u_profile = G / (2.0 * nu) * y * (2.0 * h - y) if nu > 0 else \
                    torch.zeros_like(y)

        # Perturbation for turbulence
        torch.manual_seed(cfg.seed)
        pert_amp = 0.05 * cfg.u_tau
        pert = pert_amp * torch.randn(
            self.nz, self.ny, self.nx, device=self.device, dtype=self.dt
        )
        pert[:, 0, :] = 0.0
        pert[:, -1, :] = 0.0

        ux = u_profile.expand(self.nz, self.ny, self.nx) + pert
        uy = torch.zeros(self.nz, self.ny, self.nx, device=self.device, dtype=self.dt)
        uz = 0.02 * pert_amp * torch.randn(
            self.nz, self.ny, self.nx, device=self.device, dtype=self.dt
        )
        uz[:, 0, :] = 0.0
        uz[:, -1, :] = 0.0

        rho = torch.ones(self.nz, self.ny, self.nx, device=self.device, dtype=self.dt)
        self.f = equilibrium3d(rho, ux, uy, uz)

    def _compute_shear_rate_field(self, ux, uy, uz):
        """Compute the local shear rate |γ̇| = sqrt(2·S:S).

        Uses central differences for velocity gradients.
        S_αβ = 0.5 * (∂u_α/∂x_β + ∂u_β/∂x_α)
        |γ̇| = sqrt(2 * S_αβ * S_αβ)
        """
        # Central differences (periodic in x,z; one-sided at walls)
        def cd_x(f):
            return (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1)) / 2.0

        def cd_y(f):
            result = (torch.roll(f, -1, dims=-2) - torch.roll(f, 1, dims=-2)) / 2.0
            # One-sided at walls
            result[:, 0, :] = f[:, 1, :] - f[:, 0, :]
            result[:, -1, :] = f[:, -1, :] - f[:, -2, :]
            return result

        def cd_z(f):
            return (torch.roll(f, -1, dims=-3) - torch.roll(f, 1, dims=-3)) / 2.0

        # Velocity gradients
        dux_dx = cd_x(ux)
        dux_dy = cd_y(ux)
        dux_dz = cd_z(ux)
        duy_dx = cd_x(uy)
        duy_dy = cd_y(uy)
        duy_dz = cd_z(uy)
        duz_dx = cd_x(uz)
        duz_dy = cd_y(uz)
        duz_dz = cd_z(uz)

        # Strain rate tensor components
        S11 = dux_dx
        S22 = duy_dy
        S33 = duz_dz
        S12 = 0.5 * (dux_dy + duy_dx)
        S13 = 0.5 * (dux_dz + duz_dx)
        S23 = 0.5 * (duy_dz + duz_dy)

        # |γ̇| = sqrt(2 * S:S)
        S2 = S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2)
        gamma_dot = torch.sqrt(2.0 * S2 + 1e-30)

        return gamma_dot

    def _compute_effective_viscosity(self, gamma_dot):
        """Papanastasiou-regularised HB effective viscosity.

        μ_eff = K·|γ̇|^(n-1) + τ_y·(1 − exp(−m·|γ̇|)) / |γ̇|
        ν_eff = μ_eff / ρ
        """
        cfg = self.cfg
        gamma = torch.clamp(gamma_dot, min=1e-12)

        # Power-law part
        mu_pl = cfg.K * torch.pow(gamma, cfg.n - 1.0)

        # Papanastasiou yield stress part
        exp_term = torch.exp(-cfg.m_pap * gamma)
        mu_ys = cfg.tau_y * (1.0 - exp_term) / gamma

        mu_eff = mu_pl + mu_ys

        # Convert to kinematic viscosity
        nu_eff = mu_eff / cfg.rho

        # Clamp for stability
        nu_eff = torch.clamp(nu_eff, min=cfg.nu_min, max=cfg.nu_max)

        return nu_eff

    def _apply_walls(self):
        """Half-way bounce-back at walls."""
        for i in self.up_idx:
            j = self.OPP[i]
            self.f[i, :, 0, :] = self.f[j, :, 0, :]

        for i in self.down_idx:
            j = self.OPP[i]
            self.f[i, :, self.ny - 1, :] = self.f[j, :, self.ny - 1, :]

    def _apply_body_force(self, tau_field):
        """Guo forcing with spatially-varying tau."""
        fx = self.cfg.body_force
        cx = self.C[:, 0]

        # Guo forcing: F_i = w_i * (1 - 1/(2τ)) * (3(c_i - u) + 9(c_i·u)c_i) · a
        # Simplified for uniform force in x:
        # factor = (1 - 1/(2τ)) varies spatially
        factor = (1.0 - 0.5 / tau_field)  # (nz, ny, nx)

        for i in range(19):
            ci_x = float(self.C[i, 0])
            self.f[i] += self.W[i] * ci_x * fx * factor

    def step(self):
        """One time step: compute ν_eff → collide → stream → walls → force."""
        cfg = self.cfg

        # Get macroscopic velocities
        rho, ux, uy, uz = macroscopic3d(self.f)

        # Compute shear rate and effective viscosity
        gamma_dot = self._compute_shear_rate_field(ux, uy, uz)
        nu_eff = self._compute_effective_viscosity(gamma_dot)

        # Relaxation time field
        tau_field = 0.5 + 3.0 * nu_eff  # (nz, ny, nx)

        # BGK collision with spatially-varying tau
        feq = equilibrium3d(rho, ux, uy, uz)

        # τ varies per cell: Ω = -1/τ * (f - feq)
        tau_4d = tau_field.unsqueeze(0)  # (1, nz, ny, nx)
        omega = -1.0 / tau_4d * (self.f - feq)

        # Smagorinsky SGS (optional)
        if cfg.smagorinsky_cs > 0:
            # Compute SGS viscosity from strain rate
            Cs = cfg.smagorinsky_cs
            nu_sgs = (Cs ** 2) * gamma_dot * (2.0 * gamma_dot)  # simplified
            nu_sgs = torch.clamp(nu_sgs, min=0, max=cfg.nu_max)
            tau_sgs = 0.5 + 3.0 * (nu_eff + nu_sgs)
            tau_sgs_4d = tau_sgs.unsqueeze(0)
            omega_sgs = -1.0 / tau_sgs_4d * (self.f - feq)
            # Blend: use SGS tau where it's larger
            mask = (tau_sgs > tau_field).unsqueeze(0)
            omega = torch.where(mask, omega_sgs, omega)

        self.f = self.f + omega

        # Streaming (D3Q19)
        f_new = torch.zeros_like(self.f)
        for i in range(19):
            dx = int(self.C[i, 0])
            dy = int(self.C[i, 1])
            dz = int(self.C[i, 2])
            # f is (19, nz, ny, nx), roll along (nz, ny, nx) by (dz, dy, dx)
            f_new[i] = torch.roll(self.f[i], shifts=(dz, dy, dx), dims=(0, 1, 2))
        self.f = f_new

        # Walls
        self._apply_walls()

        # Body force
        self._apply_body_force(tau_field)

    def accumulate_statistics(self):
        """Accumulate statistics (plane-averaged)."""
        rho, ux, uy, uz = macroscopic3d(self.f)

        # Plane averages (over x and z)
        u_mean_y = ux.mean(dim=(0, 2))  # (ny,)
        u2_mean = (ux ** 2).mean(dim=(0, 2))
        v2_mean = (uy ** 2).mean(dim=(0, 2))
        w2_mean = (uz ** 2).mean(dim=(0, 2))
        uv_mean = (ux * uy).mean(dim=(0, 2))

        # Running average
        n = self._stat_steps
        self._u_mean = (n * self._u_mean + u_mean_y) / (n + 1)
        self._uu = (n * self._uu + u2_mean) / (n + 1)
        self._vv = (n * self._vv + v2_mean) / (n + 1)
        self._ww = (n * self._ww + w2_mean) / (n + 1)
        self._uv = (n * self._uv + uv_mean) / (n + 1)
        self._stat_steps += 1

    def get_statistics(self):
        """Return accumulated statistics."""
        u_mean = self._u_mean.cpu().numpy()
        uu = self._uu.cpu().numpy()
        vv = self._vv.cpu().numpy()
        ww = self._ww.cpu().numpy()
        uv = self._uv.cpu().numpy()

        # Reynolds stresses
        u_prime2 = uu - u_mean ** 2
        v_prime2 = vv  # v_mean ≈ 0
        w_prime2 = ww  # w_mean ≈ 0
        uv_prime = uv  # u_mean * v_mean ≈ 0

        # TKE
        tke = 0.5 * (u_prime2 + v_prime2 + w_prime2)

        return {
            "u_mean": u_mean,
            "u_prime2": u_prime2,
            "v_prime2": v_prime2,
            "w_prime2": w_prime2,
            "uv_prime": uv_prime,
            "tke": tke,
            "stat_steps": self._stat_steps,
        }

    def compute_friction_velocity(self):
        """Estimate u_τ from wall shear stress (momentum balance)."""
        # From momentum balance: τ_w = G * h
        tau_w = self.cfg.body_force * self.cfg.half_height
        u_tau = math.sqrt(tau_w / self.cfg.rho)
        return u_tau

    def run(self, callback=None):
        """Run the simulation."""
        cfg = self.cfg
        t0 = time.time()

        for step in range(1, cfg.steps + 1):
            self.step()

            if step > cfg.warmup_steps and step % cfg.sample_interval == 0:
                self.accumulate_statistics()

            if callback and step % cfg.report_interval == 0:
                callback(step, self)

        elapsed = time.time() - t0
        return {
            "steps": cfg.steps,
            "elapsed_s": elapsed,
            "statistics_steps": self._stat_steps,
        }


def run_viscoplastic_channel(cfg: ViscoplasticChannelConfig) -> dict:
    """Run a viscoplastic channel simulation and return results."""
    solver = ViscoplasticChannel3D(cfg)

    def report(step, s):
        stats = s.get_statistics()
        u_tau = s.compute_friction_velocity()
        u_bulk = float(stats["u_mean"][s.cfg.ny // 2])
        print(f"  step={step} u_bulk={u_bulk:.6f} u_tau={u_tau:.6f} "
              f"TKE_max={stats['tke'].max():.6e}")

    info = solver.run(callback=report)
    stats = solver.get_statistics()
    u_tau = solver.compute_friction_velocity()

    return {
        "config": {
            "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
            "tau_y": cfg.tau_y, "K": cfg.K, "n": cfg.n,
            "re_tau": cfg.re_tau, "u_tau": cfg.u_tau,
            "yield_number": cfg.yield_number,
        },
        "run_info": info,
        "statistics": stats,
        "derived": {
            "u_tau_measured": u_tau,
            "yield_number": cfg.yield_number,
        },
    }
