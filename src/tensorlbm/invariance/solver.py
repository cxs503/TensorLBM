"""D2Q9 lattice-Boltzmann kernel with SDF-interpolated bounce-back (NumPy).

This is a *self-contained* solver used by the invariance verification surface.
It deliberately stays small so that every numerical ingredient that can break
grid-orientation or Galilean invariance is explicit and auditable:

``streaming``
    periodic gather (``numpy.roll`` per direction) followed by an overwrite of
    the wall-adjacent incoming populations.
``wall treatment``
    Bouzidi et al. (2001) interpolated bounce-back, ``q``-fraction taken from
    the analytic signed distance function of the body, i.e. **second-order**
    accurate for walls at arbitrary orientation.  ``bounce_back="halfway"``
    freezes ``q = 0.5`` (classic half-way bounce-back) and is used as the
    negative control of the invariance study.
``forcing``
    Guo, Zheng & Shi (2002) second-order body-force scheme (mean momentum
    absorption ``F/2`` included in the velocity used by the equilibrium).
``collision``
    BGK (default) or two-relaxation-time (TRT, magic parameter 3/16), used
    where BGK alone is too diffusive/unstable at high Reynolds number.

The implementation is pure NumPy because torch's per-op overhead (20x slower
than NumPy on the shared cluster host measured in this study) would make the
sweeps intractable; determinism (bitwise reproducibility) is also guaranteed.

References
----------
Bouzidi, M., Firdaouss, M., Lallemand, P. (2001) *Momentum transfer of a
    Boltzmann-lattice fluid with boundaries*, Phys. Fluids 13(11), 3452.
Guo, Z., Zheng, C., Shi, B. (2002) *Discrete lattice effects on the forcing
    term in the lattice Boltzmann method*, Phys. Rev. E 65, 046308.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ..d2q9 import C, OPPOSITE, W

import math

CS2 = 1.0 / 3.0
CS = math.sqrt(CS2)

CVEL = np.asarray(C.numpy(), dtype=np.int64)          # (9, 2)
OPP = np.asarray(OPPOSITE.numpy(), dtype=np.int64)     # (9,)
WEIGHTS = np.asarray(W.numpy(), dtype=np.float64)      # (9,)


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium populations with shape ``(9, ny, nx)``."""
    cx = CVEL[:, 0].reshape(9, 1, 1).astype(rho.dtype)
    cy = CVEL[:, 1].reshape(9, 1, 1).astype(rho.dtype)
    w = WEIGHTS.reshape(9, 1, 1).astype(rho.dtype)
    cu = 3.0 * (cx * ux[None] + cy * uy[None])
    usqr = 1.5 * (ux * ux + uy * uy)[None]
    return rho[None] * w * (1.0 + cu + 0.5 * cu * cu - usqr)


@dataclass
class VelocityDirichlet:
    """Velocity Dirichlet patch applied by non-equilibrium extrapolation (NEEM).

    ``shift`` is the (dy, dx) offset from the boundary node to the interior
    neighbour used for the non-equilibrium part; the equilibrium part uses the
    prescribed velocity and the neighbour density.
    """

    mask: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    shift: tuple[int, int] = (0, 1)


@dataclass
class WallData:
    q: list = field(default_factory=list)
    mask: list = field(default_factory=list)


class D2Q9Solver:
    """Minimal, auditable D2Q9 solver for invariance studies."""

    def __init__(
        self,
        geom,
        tau: float = 0.8,
        *,
        dtype=np.float64,
        collision: str = "bgk",
        trt_magic: float = 3.0 / 16.0,
        rho0: float = 1.0,
        bounce_back: str = "interpolated",
        device: str | None = None,   # accepted for API symmetry; NumPy only
    ) -> None:
        if collision not in ("bgk", "trt"):
            raise ValueError(f"unknown collision {collision!r}")
        if bounce_back not in ("interpolated", "halfway"):
            raise ValueError(f"unknown bounce_back {bounce_back!r}")
        self.geom = geom
        self.nx, self.ny = geom.nx, geom.ny
        self.tau = float(tau)
        self.omega = 1.0 / self.tau
        self.collision_mode = collision
        self.bounce_back = bounce_back
        self.dtype = dtype
        self.rho0 = float(rho0)
        if collision == "trt":
            self.tau_minus = 0.5 + trt_magic / (self.tau - 0.5)
            self.omega_minus = 1.0 / self.tau_minus

        self.solid = np.ascontiguousarray(geom.solid)
        self.fluid = ~self.solid
        self.rho = np.full((self.ny, self.nx), self.rho0, dtype=dtype)
        self.u = np.zeros((2, self.ny, self.nx), dtype=dtype)
        self.f = equilibrium(
            self.rho, self.u[0], self.u[1]
        )
        self.dirs: list[VelocityDirichlet] = []
        self._walls = self._build_walls()
        zeros = np.zeros_like(self.rho)
        self._rest_eq = equilibrium(self.rho, zeros, zeros)

    # ------------------------------------------------------------------ setup
    def _build_walls(self) -> WallData:
        sdf = self.geom.sdf.astype(self.dtype)
        gx = self.geom.gx.astype(self.dtype)
        gy = self.geom.gy.astype(self.dtype)
        walls = WallData()
        for i in range(9):
            cx, cy = int(CVEL[i, 0]), int(CVEL[i, 1])
            if cx == 0 and cy == 0:
                walls.q.append(None)
                walls.mask.append(None)
                continue
            nb_solid = np.roll(self.solid, shift=(-cy, -cx), axis=(0, 1))
            mask = self.fluid & nb_solid
            if self.bounce_back == "halfway":
                q = np.full_like(sdf, 0.5)
            else:
                gn = gx * cx + gy * cy
                gn = np.where(np.abs(gn) > 1e-12, gn, -1e-12)
                q = np.clip(-sdf / gn, 1e-4, 1.0).astype(self.dtype)
            walls.q.append(np.clip(q, 1e-4, 1.0).astype(self.dtype))
            walls.mask.append(mask)
        return walls

    def set_velocity_dirichlet(self, bcs: list[VelocityDirichlet]) -> None:
        self.dirs = [
            VelocityDirichlet(
                mask=np.asarray(bc.mask, dtype=bool),
                ux=bc.ux.astype(self.dtype),
                uy=bc.uy.astype(self.dtype),
                shift=bc.shift,
            )
            for bc in bcs
        ]

    def initialise(self, ux, uy, rho=None) -> None:
        uxt = np.asarray(ux, dtype=self.dtype)
        uyt = np.asarray(uy, dtype=self.dtype)
        rt = (np.full((self.ny, self.nx), self.rho0, dtype=self.dtype)
              if rho is None else np.asarray(rho, dtype=self.dtype))
        self.f = equilibrium(rt, uxt, uyt)
        # solid nodes hold the resting equilibrium from the start: a zero
        # population would make rho = 0 there and poison the momentum/velocity
        # divisions (0/0) in the first collision step.
        self.f[:, self.solid] = self._rest_eq[:, self.solid]

    # ------------------------------------------------------------ primitives
    def macroscopic(self, f: np.ndarray | None = None):
        f = self.f if f is None else f
        rho = f.sum(0)
        rho_safe = np.where(rho > 1e-12, rho, 1.0)
        ux = np.tensordot(CVEL[:, 0].astype(self.dtype), f, axes=(0, 0)) / rho_safe
        uy = np.tensordot(CVEL[:, 1].astype(self.dtype), f, axes=(0, 0)) / rho_safe
        return rho, ux, uy

    def collide(self, f: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        cx = CVEL[:, 0].astype(self.dtype).reshape(9, 1, 1)
        cy = CVEL[:, 1].astype(self.dtype).reshape(9, 1, 1)
        w = WEIGHTS.astype(self.dtype).reshape(9, 1, 1)
        rho = f.sum(0)
        rho_safe = np.where(rho > 1e-12, rho, 1.0)
        momx = np.tensordot(CVEL[:, 0].astype(self.dtype), f, axes=(0, 0))
        momy = np.tensordot(CVEL[:, 1].astype(self.dtype), f, axes=(0, 0))
        ux = (momx + 0.5 * fx) / rho_safe
        uy = (momy + 0.5 * fy) / rho_safe
        feq = equilibrium(rho, ux, uy)
        cu = cx * ux[None] + cy * uy[None]
        xi = (cx - ux[None]) * fx[None] + (cy - uy[None]) * fy[None]
        src = (1.0 - 0.5 * self.omega) * w * (3.0 * xi + 9.0 * cu * (cx * fx[None] + cy * fy[None]))
        if self.collision_mode == "bgk":
            return f - self.omega * (f - feq) + src
        f_opp = f[OPP]
        feq_opp = feq[OPP]
        src_opp = src[OPP]
        out = (0.5 * (f + f_opp) - self.omega * 0.5 * (f + f_opp - feq - feq_opp) + 0.5 * (src + src_opp))
        out += (0.5 * (f - f_opp) - self.omega_minus * 0.5 * (f - f_opp - feq + feq_opp)
                + 0.5 * (src - src_opp))
        return out

    def stream_interpolated(self, fs: np.ndarray) -> np.ndarray:
        f_new = np.empty_like(fs)
        for i in range(9):
            f_new[i] = np.roll(fs[i], shift=(int(CVEL[i, 1]), int(CVEL[i, 0])), axis=(0, 1))
        walls = self._walls
        for i in range(1, 9):
            mask = walls.mask[i]
            if mask is None or not mask.any():
                continue
            cx, cy = int(CVEL[i, 0]), int(CVEL[i, 1])
            q = walls.q[i]
            # post-collision value at the boundary node, direction i (into wall)
            fi_b = fs[i]
            # node one lattice step further inside the fluid: xb - c_i
            fi_2 = np.roll(fs[i], shift=(cy, cx), axis=(0, 1))
            fopp_2 = np.roll(fs[OPP[i]], shift=(cy, cx), axis=(0, 1))
            near = q < 0.5
            val_near = 2.0 * q * fi_b + (1.0 - 2.0 * q) * fi_2
            inv = 1.0 / (2.0 * q)
            val_far = inv * fi_b + (1.0 - inv) * fopp_2
            val = np.where(near, val_near, val_far)
            np.copyto(f_new[OPP[i]], val, where=mask)
        return f_new

    def apply_velocity_dirichlet(self, f: np.ndarray) -> np.ndarray:
        if not self.dirs:
            return f
        for bc in self.dirs:
            dy, dx = bc.shift
            sh = (-dy, -dx)
            rho_n = np.roll(f.sum(0), shift=sh, axis=(0, 1))
            rho_n = np.where(rho_n > 1e-12, rho_n, 1.0)
            mox = np.roll(np.tensordot(CVEL[:, 0].astype(self.dtype), f, axes=(0, 0)), shift=sh, axis=(0, 1))
            moy = np.roll(np.tensordot(CVEL[:, 1].astype(self.dtype), f, axes=(0, 0)), shift=sh, axis=(0, 1))
            uxn = mox / rho_n
            uyn = moy / rho_n
            f_nb = np.roll(f, shift=(sh[0], sh[1]), axis=(1, 2))
            feq_nb = equilibrium(rho_n, uxn, uyn)
            feq_bc = equilibrium(rho_n, bc.ux, bc.uy)
            val = feq_bc + (f_nb - feq_nb)
            f[:, bc.mask] = val[:, bc.mask]
        return f

    def step(self, fx: float | np.ndarray = 0.0, fy: float | np.ndarray = 0.0) -> None:
        if not isinstance(fx, np.ndarray):
            fx = np.full((self.ny, self.nx), float(fx), dtype=self.dtype)
        if not isinstance(fy, np.ndarray):
            fy = np.full((self.ny, self.nx), float(fy), dtype=self.dtype)
        fs = self.collide(self.f, fx, fy)
        f_new = self.stream_interpolated(fs)
        f_new = self.apply_velocity_dirichlet(f_new)
        f_new[:, self.solid] = self._rest_eq[:, self.solid]
        self.f = f_new

    # ------------------------------------------------------------ diagnostics
    def velocity(self):
        _, ux, uy = self.macroscopic()
        return ux, uy

    def state_hash(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.f).tobytes()).hexdigest()

    def kinetic_energy(self) -> float:
        _, ux, uy = self.macroscopic()
        return float(0.5 * ((ux * ux + uy * uy) * self.fluid).sum())

    def momentum(self):
        """Total fluid momentum ``∫ ρ u`` per unit depth (lattice units)."""
        rho, ux, uy = self.macroscopic()
        m = self.fluid
        return float((rho * ux * m).sum()), float((rho * uy * m).sum())