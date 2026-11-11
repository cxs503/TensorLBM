"""3D square duct Poiseuille flow solver using D3Q19 LBM.

Follows the same pattern as the existing D2Q9 channel solver (lbm_channel.py).

Reference: Shah & London (1978), f·Re = 56.908 for square duct.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


# D3Q19 lattice
CX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=int)
CY = np.array([0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=int)
CZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1], dtype=int)
WT = np.array([1/3] + [1/18]*6 + [1/36]*12, dtype=float)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=int)

# Mirror directions for specular reflection (not used for no-slip, but needed for table)
MIRROR_Y = np.array([0, 1, 2, 4, 3, 5, 6, 8, 7, 10, 9, 11, 12, 14, 13, 16, 15, 18, 17], dtype=int)
MIRROR_Z = np.array([0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 12, 11, 13, 14, 15, 16, 17, 18], dtype=int)


@dataclass(frozen=True)
class SquareDuctConfig:
    nx: int = 8
    ny: int = 32
    nz: int = 32
    tau: float = 1.0
    force: float = 1e-5
    steps: int = 20000
    warmup: int = 5000
    sample_interval: int = 100
    seed: int = 42
    label: str = "square_duct"

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def hydraulic_diameter(self) -> float:
        return float(self.ny - 2)

    def validate(self) -> None:
        if self.ny < 8 or self.nz < 8:
            raise ValueError("Cross-section too small")
        if self.nx < 4:
            raise ValueError("nx too small")
        if self.tau <= 0.5:
            raise ValueError("tau must be > 0.5")
        if self.warmup >= self.steps:
            raise ValueError("warmup must be < steps")


class SquareDuctSolver:
    """D3Q19 BGK solver for square duct with no-slip walls and Guo forcing."""

    def __init__(self, cfg: SquareDuctConfig):
        cfg.validate()
        self.cfg = cfg
        self.nx, self.ny, self.nz = cfg.nx, cfg.ny, cfg.nz
        self.tau = cfg.tau
        self.nu = cfg.nu
        self.force = cfg.force
        self.omega = 1.0 / self.tau

        # Solid mask: walls at y=0, y=ny-1, z=0, z=nz-1
        self.solid = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        self.solid[:, 0, :] = True
        self.solid[:, -1, :] = True
        self.solid[0, :, :] = True
        self.solid[-1, :, :] = True
        self.fluid = ~self.solid

        self._build_reflection_table()
        self._init_populations()

    def _build_reflection_table(self) -> None:
        """Build bounce-back reflection table (same pattern as D2Q9 channel)."""
        table = []
        for i in range(1, 19):
            cx, cy, cz = CX[i], CY[i], CZ[i]
            # Compute source solid mask (where populations come from after streaming)
            src = self.solid.copy()
            if cz != 0:
                src = np.roll(src, -cz, axis=0)
            if cy != 0:
                src = np.roll(src, -cy, axis=1)
            if cx != 0:
                src = np.roll(src, -cx, axis=2)
            # Mask: fluid nodes receiving from solid
            m = src & self.fluid
            if not m.any():
                continue
            table.append((i, OPP[i], m))
        self._reflection = table

    def _init_populations(self) -> None:
        cfg = self.cfg
        ny, nz, nx = self.ny, self.nz, self.nx

        y = np.arange(ny).reshape(1, ny, 1)
        z = np.arange(nz).reshape(nz, 1, 1)
        h_y = (ny - 2) / 2.0
        h_z = (nz - 2) / 2.0
        y_c = y - (ny - 1) / 2.0
        z_c = z - (nz - 1) / 2.0

        ux = cfg.force / (2.0 * self.nu) * (h_y**2 - y_c**2) * (h_z**2 - z_c**2) / max(h_y**2, h_z**2)
        ux = np.broadcast_to(ux, (nz, ny, nx)).copy()
        ux[:, 0, :] = 0.0; ux[:, -1, :] = 0.0
        ux[0, :, :] = 0.0; ux[-1, :, :] = 0.0

        np.random.seed(cfg.seed)
        pert = 0.01 * cfg.force * np.random.randn(nz, ny, nx)
        pert[:, 0, :] = 0.0; pert[:, -1, :] = 0.0
        pert[0, :, :] = 0.0; pert[-1, :, :] = 0.0
        ux += pert

        uy = np.zeros((nz, ny, nx))
        uz = np.zeros((nz, ny, nx))
        rho = np.ones((nz, ny, nx))

        self.f = self._equilibrium(rho, ux, uy, uz)

    def _equilibrium(self, rho, ux, uy, uz):
        feq = np.empty((19, self.nz, self.ny, self.nx))
        u2 = ux**2 + uy**2 + uz**2
        for i in range(19):
            cu = 3.0 * (CX[i] * ux + CY[i] * uy + CZ[i] * uz)
            feq[i] = WT[i] * rho * (1.0 + cu + 0.5 * cu**2 - 1.5 * u2)
        return feq

    def _macroscopic(self, f):
        rho = f.sum(axis=0)
        ux = (f * CX.reshape(-1, 1, 1, 1)).sum(axis=0) / rho
        uy = (f * CY.reshape(-1, 1, 1, 1)).sum(axis=0) / rho
        uz = (f * CZ.reshape(-1, 1, 1, 1)).sum(axis=0) / rho
        return rho, ux, uy, uz

    def _stream(self) -> np.ndarray:
        """Stream + bounce-back (same pattern as D2Q9 channel)."""
        f = self.f
        out = np.empty_like(f)
        out[0] = f[0]
        for i in range(1, 19):
            cx, cy, cz = CX[i], CY[i], CZ[i]
            fi = f[i]
            if cz != 0:
                fi = np.roll(fi, cz, axis=0)
            if cy != 0:
                fi = np.roll(fi, cy, axis=1)
            if cx != 0:
                fi = np.roll(fi, cx, axis=2)
            out[i] = fi
        # Bounce-back: use PRE-streaming populations
        for i, jj, mask in self._reflection:
            out[jj][mask] = f[i][mask]
        return out

    def _collide_and_force(self) -> np.ndarray:
        rho, ux, uy, uz = self._macroscopic(self.f)
        feq = self._equilibrium(rho, ux, uy, uz)
        f_post = self.f - self.omega * (self.f - feq)
        # Guo forcing
        gf = (1.0 - 0.5 * self.omega) * 3.0 * self.force
        for i in range(19):
            f_post[i] += gf * WT[i] * (CX[i] - ux)
        return f_post

    def step(self) -> None:
        f_post = self._collide_and_force()
        self.f = f_post
        self.f = self._stream()

    def get_mean_velocity(self) -> float:
        _, ux, _, _ = self._macroscopic(self.f)
        return float(ux[:, 1:-1, 1:-1].mean())

    def get_max_velocity(self) -> float:
        _, ux, _, _ = self._macroscopic(self.f)
        return float(ux[:, 1:-1, 1:-1].max())

    def get_velocity_profile(self):
        _, ux, _, _ = self._macroscopic(self.f)
        z_mid, y_mid = self.nz // 2, self.ny // 2
        return (np.arange(self.ny), ux[z_mid, :, 0].copy(),
                np.arange(self.nz), ux[:, y_mid, 0].copy())

    def get_velocity_field(self) -> np.ndarray:
        _, ux, _, _ = self._macroscopic(self.f)
        return ux[:, :, 0].copy()

    def run(self, callback=None) -> dict:
        cfg = self.cfg
        t0 = time.time()
        u_history, u_acc, n_acc = [], 0.0, 0
        for step in range(1, cfg.steps + 1):
            self.step()
            if step >= cfg.warmup and step % cfg.sample_interval == 0:
                u = self.get_mean_velocity()
                u_acc += u; n_acc += 1
                u_history.append((step, u))
            if callback and step % 1000 == 0:
                callback(step, self)
        elapsed = time.time() - t0
        return {
            "steps": cfg.steps, "warmup": cfg.warmup,
            "elapsed_s": elapsed, "steps_per_sec": cfg.steps / max(elapsed, 1e-9),
            "u_mean": u_acc / max(n_acc, 1), "u_max": self.get_max_velocity(),
            "u_history": u_history, "n_samples": n_acc,
        }


def compute_fRe(force, nu, u_mean, D_h):
    return force * D_h**2 / (2.0 * nu * u_mean)


def run_square_duct(cfg: SquareDuctConfig) -> dict:
    solver = SquareDuctSolver(cfg)
    run_info = solver.run()
    D_h = cfg.hydraulic_diameter
    fRe = compute_fRe(cfg.force, cfg.nu, run_info["u_mean"], D_h)
    y_coords, uy_profile, z_coords, uz_profile = solver.get_velocity_profile()
    u_field = solver.get_velocity_field()
    return {
        "config": cfg, "run_info": run_info, "fRe": fRe, "D_h": D_h,
        "y_coords": y_coords, "uy_profile": uy_profile,
        "z_coords": z_coords, "uz_profile": uz_profile,
        "u_field": u_field, "solver": solver,
    }
