"""Fast LES channel verification using optimized numpy-based streaming.

This module provides a faster implementation suitable for verification tests.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class LESChannelConfig:
    """Configuration for LES channel flow simulation."""
    nx: int = 32
    ny: int = 17
    nz: int = 16
    half_height: float = 8.0
    re_tau: float = 180.0
    u_tau: float = 0.01
    smagorinsky_cs: float = 0.12
    steps: int = 5000
    warmup_steps: int = 1000
    sample_interval: int = 10
    seed: int = 42

    def __post_init__(self):
        self.nu = self.u_tau * self.half_height / self.re_tau

    @property
    def body_force(self) -> float:
        H = 2.0 * self.half_height
        return 2.0 * self.u_tau**2 / H


class FastLESChannelSolver:
    """Fast numpy-based LES channel solver for verification."""

    def __init__(self, cfg: LESChannelConfig):
        self.cfg = cfg
        self.nx, self.ny, self.nz = cfg.nx, cfg.ny, cfg.nz
        self.nu = cfg.nu
        self.tau = 0.5 + 3.0 * cfg.nu
        
        # D3Q19 constants
        self._init_lattice()
        self._init_populations()

    def _init_lattice(self):
        """Initialize D3Q19 lattice."""
        # Lattice velocities
        self.c = np.array([
            [0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
            [1,1,0], [-1,-1,0], [1,-1,0], [-1,1,0], [1,0,1], [-1,0,-1],
            [1,0,-1], [-1,0,1], [0,1,1], [0,-1,-1], [0,1,-1], [0,-1,1]
        ], dtype=np.int32)
        
        # Weights
        self.w = np.array([1/3] + [1/18]*6 + [1/36]*12, dtype=np.float64)
        
        # Opposite directions
        self.opp = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17])

    def _init_populations(self):
        """Initialize with parabolic profile."""
        cfg = self.cfg
        h = cfg.half_height
        a = cfg.body_force
        
        y = np.arange(self.ny).reshape(1, self.ny, 1)
        u_mean = a * y * (2*h - y) / (2.0 * self.nu)
        
        np.random.seed(cfg.seed)
        pert = 0.1 * cfg.u_tau * np.random.randn(self.nz, self.ny, self.nx)
        pert[:, 0, :] = 0
        pert[:, -1, :] = 0
        
        ux = np.broadcast_to(u_mean, (self.nz, self.ny, self.nx)).copy() + pert
        uy = np.zeros((self.nz, self.ny, self.nx))
        uz = 0.1 * 0.1 * cfg.u_tau * np.random.randn(self.nz, self.ny, self.nx)
        uz[:, 0, :] = 0
        uz[:, -1, :] = 0
        
        rho = np.ones((self.nz, self.ny, self.nx))
        self.f = self._equilibrium(rho, ux, uy, uz)

    def _equilibrium(self, rho, ux, uy, uz):
        """Compute equilibrium."""
        feq = np.zeros((19, self.nz, self.ny, self.nx))
        u2 = ux**2 + uy**2 + uz**2
        
        for i in range(19):
            uc = self.c[i,0]*ux + self.c[i,1]*uy + self.c[i,2]*uz
            feq[i] = self.w[i] * rho * (1 + 3*uc + 4.5*uc**2 - 1.5*u2)
        
        return feq

    def _macroscopic(self):
        """Compute macroscopic quantities."""
        rho = self.f.sum(axis=0)
        ux = np.zeros_like(rho)
        uy = np.zeros_like(rho)
        uz = np.zeros_like(rho)
        
        for i in range(19):
            ux += self.c[i,0] * self.f[i]
            uy += self.c[i,1] * self.f[i]
            uz += self.c[i,2] * self.f[i]
        
        return rho, ux/rho, uy/rho, uz/rho

    def _stream(self):
        """Stream with periodic BC."""
        f_new = np.empty_like(self.f)
        for i in range(19):
            cx, cy, cz = self.c[i]
            f_new[i] = np.roll(self.f[i], shift=(-cz, -cy, -cx), axis=(0,1,2))
        self.f = f_new

    def _apply_walls(self):
        """Bounce-back at walls."""
        for i in range(19):
            if self.c[i,1] > 0:  # going up
                self.f[i, :, 0, :] = self.f[self.opp[i], :, 0, :]
            elif self.c[i,1] < 0:  # going down
                self.f[i, :, self.ny-1, :] = self.f[self.opp[i], :, self.ny-1, :]

    def _collide_smagorinsky(self):
        """Smagorinsky collision."""
        rho, ux, uy, uz = self._macroscopic()
        feq = self._equilibrium(rho, ux, uy, uz)
        fneq = self.f - feq
        
        # Compute non-equilibrium stress
        Pi_xx = np.sum([fneq[i] * self.c[i,0]**2 for i in range(19)], axis=0)
        Pi_yy = np.sum([fneq[i] * self.c[i,1]**2 for i in range(19)], axis=0)
        Pi_zz = np.sum([fneq[i] * self.c[i,2]**2 for i in range(19)], axis=0)
        Pi_xy = np.sum([fneq[i] * self.c[i,0]*self.c[i,1] for i in range(19)], axis=0)
        Pi_xz = np.sum([fneq[i] * self.c[i,0]*self.c[i,2] for i in range(19)], axis=0)
        Pi_yz = np.sum([fneq[i] * self.c[i,1]*self.c[i,2] for i in range(19)], axis=0)
        
        Pi_Q = np.sqrt(2*(Pi_xx**2 + Pi_yy**2 + Pi_zz**2 + 
                          2*Pi_xy**2 + 2*Pi_xz**2 + 2*Pi_yz**2) + 1e-20)
        
        # Smagorinsky tau_eff
        Cs = self.cfg.smagorinsky_cs
        Q = 18 * Cs**2 * Pi_Q / rho
        tau_eff = 0.5 * (self.tau + np.sqrt(self.tau**2 + 3*Q))
        tau_eff = np.clip(tau_eff, 0.5001, 2.0)
        
        omega = 1.0 / tau_eff
        self.f = self.f - omega[np.newaxis] * (self.f - feq)

    def _apply_force(self):
        """Apply body force."""
        fx = self.cfg.body_force
        factor = 1.0 - 0.5/self.tau
        
        for i in range(19):
            self.f[i] += self.w[i] * self.c[i,0] * fx * factor

    def step(self):
        """One time step."""
        self._collide_smagorinsky()
        self._stream()
        self._apply_walls()
        self._apply_force()


def run_fast_case(
    half_height: int = 8,
    steps: int = 5000,
    warmup_steps: int = 1000,
    u_tau: float = 0.01,
    re_tau: float = 180.0,
    cs: float = 0.12,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run fast LES channel verification."""
    
    cfg = LESChannelConfig(
        half_height=half_height,
        re_tau=re_tau,
        u_tau=u_tau,
        smagorinsky_cs=cs,
        steps=steps,
        warmup_steps=warmup_steps,
    )
    
    if verbose:
        print(f"Fast LES Channel: Re_tau={re_tau}, h={half_height}")
        print(f"Grid: {cfg.nx}x{cfg.ny}x{cfg.nz}")
        print(f"Cs={cs}, nu={cfg.nu:.6f}")
    
    solver = FastLESChannelSolver(cfg)
    
    # Statistics
    ny = cfg.ny
    U_sum = np.zeros(ny)
    u2_sum = np.zeros(ny)
    v2_sum = np.zeros(ny)
    w2_sum = np.zeros(ny)
    uv_sum = np.zeros(ny)
    n_samples = 0
    
    t0 = time.time()
    for step in range(1, steps + 1):
        solver.step()
        
        if step > warmup_steps and step % cfg.sample_interval == 0:
            rho, ux, uy, uz = solver._macroscopic()
            
            u_xz = ux.mean(axis=(0, 2))
            v_xz = uy.mean(axis=(0, 2))
            w_xz = uz.mean(axis=(0, 2))
            
            n_samples += 1
            alpha = 1.0 / n_samples
            U_sum = (1-alpha)*U_sum + alpha*u_xz
            
            u_prime = ux - u_xz.reshape(1,-1,1)
            v_prime = uy - v_xz.reshape(1,-1,1)
            w_prime = uz - w_xz.reshape(1,-1,1)
            
            u2_sum = (1-alpha)*u2_sum + alpha*(u_prime**2).mean(axis=(0,2))
            v2_sum = (1-alpha)*v2_sum + alpha*(v_prime**2).mean(axis=(0,2))
            w2_sum = (1-alpha)*w2_sum + alpha*(w_prime**2).mean(axis=(0,2))
            uv_sum = (1-alpha)*uv_sum + alpha*(u_prime*v_prime).mean(axis=(0,2))
        
        if verbose and step % 1000 == 0:
            rho, ux, uy, uz = solver._macroscopic()
            print(f"Step {step}/{steps}, u_bulk={ux.mean():.5f}")
    
    elapsed = time.time() - t0
    
    # Compute profiles
    y = np.arange(ny)
    h = half_height
    y_from_wall = np.minimum(y, ny-1-y)
    dy = 2.0*h / (ny-1)
    y_plus = y_from_wall * dy * u_tau / cfg.nu
    
    U_plus = U_sum / u_tau
    u_rms_plus = np.sqrt(u2_sum) / u_tau
    v_rms_plus = np.sqrt(v2_sum) / u_tau
    w_rms_plus = np.sqrt(w2_sum) / u_tau
    uv_plus = -uv_sum / u_tau**2
    
    # Build artifact
    artifact = {
        "label": f"re{int(re_tau)}_h{half_height}_fast",
        "config": {
            "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
            "half_height": half_height,
            "re_tau": re_tau,
            "u_tau": u_tau,
            "cs": cs,
            "steps": steps,
            "warmup_steps": warmup_steps,
        },
        "profiles": {
            "y_plus": y_plus.tolist(),
            "U_plus": U_plus.tolist(),
            "u_rms_plus": u_rms_plus.tolist(),
            "v_rms_plus": v_rms_plus.tolist(),
            "w_rms_plus": w_rms_plus.tolist(),
            "uv_plus": uv_plus.tolist(),
        },
        "summary": {
            "n_samples": n_samples,
            "elapsed_s": elapsed,
        },
    }
    
    return artifact


if __name__ == "__main__":
    artifact = run_fast_case(half_height=8, steps=2000, warmup_steps=500, verbose=True)
    print(f"\nElapsed: {artifact['summary']['elapsed_s']:.1f}s")
    print(f"Samples: {artifact['summary']['n_samples']}")
