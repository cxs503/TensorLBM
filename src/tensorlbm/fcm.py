"""Force-coupling method (FCM) for particle-laden LBM flows.

This module implements the force-coupling framework of Maxey & Patel (2004) for
resolved spherical particles on top of the D3Q19 lattice-Boltzmann solver used
elsewhere in :mod:`tensorlbm`.

Method
------
FCM represents the momentum coupling between a particle and the fluid by an
isotropic Gaussian force envelope,

    f(x,t) = sum_p sum_{m in p} F_m * delta_sigma(x - X_m),
    delta_sigma(r) = (1/(sqrt(pi) sigma)^3) exp(-r^2/sigma^2),

and recovers the fluid velocity at the coupling points with the *adjoint* of the
same kernel.  The discrete weights are renormalised per coupling point, so that
``sum_x delta = 1`` (exact force conservation) and partition of unity (exact
interpolation) hold; the adjointness identity
``sum_x f(x).u(x) = sum_m F_m . u(X_m)`` is asserted in the unit tests.

Textbook FCM applies the envelope to a *single* force monopole per particle,
which is appropriate for point-like particles (``a << sigma``).  For resolved
particles (``a >= 1.5 dx``) a monopole cannot enforce the rigid-body constraint
over the particle volume: the centre velocity is zeroed while the rest of the
envelope slips, and the measured reaction carries no drag information.  The
implementation here therefore applies the envelope to a cloud of coupling
points on the particle surface -- the discrete analogue of the
force + torque + stresslet multipole expansion of Maxey & Patel (2004) -- with
the direct-forcing form

    F_m = (rho_f dx^3/dt) (U_p + omega_p x r_m - u(X_m)),

so that ``-sum_m F_m`` is the hydrodynamic force on the particle:

    m_p dU_p/dt = F_ext,p - sum_m F_m,      m_p = rho_p V_p.

The per-step relaxation factor is ``rho_f dx^3/m_p = O(1/N_m)``, hence the
scheme is stable for any density ratio (no added-mass instability).

References
----------
Maxey, M. R. & Patel, B. K. (2004). "Localized force representations for
    particles sedimenting in Stokes flow." *Int. J. Multiphase Flow* 27, 1603.
Roma, A. M., Peskin, C. S. & Berger, S. A. (1999). *J. Comput. Phys.* 153, 509.
Uhlmann, M. (2005). "An immersed boundary method with direct forcing for the
    simulation of particulate flows." *J. Comput. Phys.* 209, 448.
Guo, Z., Zheng, C. & Shi, B. (2002). *Phys. Rev. E* 65, 046308.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .d3q19 import C as C3D19
from .d3q19 import W as W3D19
from .d3q19 import equilibrium3d, macroscopic3d
from .solver3d import stream3d

__all__ = [
    "FCMCoupler",
    "FCMSimulation",
    "ParticleSet",
    "gaussian_delta_1d",
    "lbm_step_bgk_guo",
    "sphere_markers",
    "sphere_particle",
]


# --------------------------------------------------------------------------- #
# Gaussian envelope
# --------------------------------------------------------------------------- #


def gaussian_delta_1d(r: torch.Tensor, sigma: float) -> torch.Tensor:
    """Normalised 1-D Gaussian kernel ``delta_sigma(r)`` (Roma et al. 1999)."""
    return torch.exp(-((r / sigma) ** 2)) / (math.sqrt(math.pi) * sigma)


def _weights_1d(dx: torch.Tensor, sigma: float) -> torch.Tensor:
    """Discretely renormalised 1-D Gaussian weights along the last axis."""
    w = gaussian_delta_1d(dx, sigma)
    return w / w.sum(dim=-1, keepdim=True)


# --------------------------------------------------------------------------- #
# Coupler: spread / interpolate on a periodic lattice
# --------------------------------------------------------------------------- #


class FCMCoupler:
    """Gaussian force envelope (spread) and matching interpolation operator.

    Args:
        shape: Eulerian grid shape ``(nz, ny, nx)`` (unit spacing, periodic).
        sigma: envelope width in lattice units.
        support_factor: support radius in units of ``sigma`` (default 2.0, where
            the Gaussian has decayed to ~2e-4 of its peak).
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        sigma: float,
        *,
        support_factor: float = 2.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        if any(n <= 0 for n in shape):
            raise ValueError(f"invalid grid shape {shape}")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.shape = tuple(int(n) for n in shape)
        self.sigma = float(sigma)
        self.dtype = dtype
        self.device = torch.device(device)
        m = int(math.ceil(support_factor * self.sigma))
        self.half_support = max(m, 1)
        self.offsets = torch.arange(
            -self.half_support, self.half_support + 1, device=self.device, dtype=dtype
        )
        self.support = int(self.offsets.numel())
        self._cache = None
        self._cache_positions = None

    def _grid_size(self) -> int:
        nz, ny, nx = self.shape
        return nz * ny * nx

    def _build(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nz, ny, nx = self.shape
        pos = positions.to(self.dtype)
        n = pos.shape[0]
        s = self.support
        offs = self.offsets
        base = torch.floor(pos)
        ix = base[:, 0].unsqueeze(1) + offs.unsqueeze(0)
        iy = base[:, 1].unsqueeze(1) + offs.unsqueeze(0)
        iz = base[:, 2].unsqueeze(1) + offs.unsqueeze(0)
        wx = _weights_1d(ix - pos[:, 0].unsqueeze(1), self.sigma)
        wy = _weights_1d(iy - pos[:, 1].unsqueeze(1), self.sigma)
        wz = _weights_1d(iz - pos[:, 2].unsqueeze(1), self.sigma)
        w3 = wx.view(n, s, 1, 1) * wy.view(n, 1, s, 1) * wz.view(n, 1, 1, s)
        w3 = w3 / w3.reshape(n, -1).sum(dim=1).view(n, 1, 1, 1)
        idx = (
            (iz.to(torch.int64) % nz).view(n, s, 1, 1) * (ny * nx)
            + (iy.to(torch.int64) % ny).view(n, 1, s, 1) * nx
            + (ix.to(torch.int64) % nx).view(n, 1, 1, s)
        ).expand(n, s, s, s)
        return idx.reshape(n, -1), w3.reshape(n, -1)

    def _get(self, positions: torch.Tensor):
        if (
            self._cache is not None
            and self._cache_positions is not None
            and self._cache_positions.shape == positions.shape
            and torch.equal(self._cache_positions, positions)
        ):
            return self._cache
        idx, w = self._build(positions)
        self._cache = (idx, w)
        self._cache_positions = positions.clone()
        return idx, w

    def spread(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Scatter per-coupling-point forces ``(M, 3)`` onto a ``(3, nz, ny, nx)`` grid."""
        n = values.shape[0]
        nz, ny, nx = self.shape
        grid = torch.zeros(self._grid_size(), 3, dtype=self.dtype, device=self.device)
        if n == 0:
            return grid.reshape(nz, ny, nx, 3).permute(3, 0, 1, 2)
        idx, w = self._get(positions)
        contrib = (w.unsqueeze(-1) * values.to(self.dtype).unsqueeze(1)).reshape(-1, 3)
        grid.index_add_(0, idx.reshape(-1), contrib)
        return grid.reshape(nz, ny, nx, 3).permute(3, 0, 1, 2)

    def interpolate(self, fields: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Interpolate ``(3, nz, ny, nx)`` at ``(M, 3)`` points (adjoint operator)."""
        n = positions.shape[0]
        if n == 0:
            return torch.zeros(0, 3, dtype=self.dtype, device=self.device)
        idx, w = self._get(positions)
        flat = fields.to(self.dtype).reshape(3, -1)
        vals = flat[:, idx.reshape(-1)].reshape(3, n, -1)
        return (vals * w.unsqueeze(0)).sum(dim=2).transpose(0, 1)


# --------------------------------------------------------------------------- #
# Particles and their coupling-point clouds
# --------------------------------------------------------------------------- #


def sphere_markers(radius: float, spacing: float = 1.0) -> torch.Tensor:
    """Equal-area spiral point set on a sphere, relative to its centre.

    Args:
        radius: sphere radius in lattice units.
        spacing: target point spacing in lattice units (``<= 1.2`` recommended).

    Returns:
        ``(M, 3)`` float32 tensor of offsets from the centre.
    """
    area = 4.0 * math.pi * radius**2
    n = max(6, int(round(area / (spacing**2))))
    i = torch.arange(n, dtype=torch.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    pts = torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1) * radius
    return pts.to(torch.float32)


@dataclass
class ParticleSet:
    """Spherical particles driven by FCM.

    Attributes:
        positions: ``(N, 3)`` centres in lattice coordinates.
        radii: ``(N,)`` radii in lattice units.
        velocities: ``(N, 3)`` particle velocities.
        density_ratio: ``(N,)`` (or scalar) ``rho_p / rho_f``.
        fixed: ``(N,)`` bool; ``True`` keeps the velocity prescribed.
    """

    positions: torch.Tensor
    radii: torch.Tensor
    velocities: torch.Tensor
    density_ratio: torch.Tensor
    fixed: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    def volumes(self) -> torch.Tensor:
        return (4.0 / 3.0) * math.pi * self.radii**3

    def masses(self, rho0: float = 1.0) -> torch.Tensor:
        return rho0 * self.density_ratio.view(-1) * self.volumes()


def sphere_particle(
    position, radius: float, *, density_ratio: float = 1.0, velocity=(0.0, 0.0, 0.0),
    fixed: bool = False, dtype=torch.float32, device="cpu",
) -> ParticleSet:
    """Single-sphere convenience constructor."""
    pos = torch.tensor([position], dtype=dtype, device=device).reshape(1, 3)
    vel = torch.tensor([velocity], dtype=dtype, device=device).reshape(1, 3)
    return ParticleSet(
        positions=pos,
        radii=torch.full((1,), float(radius), dtype=dtype, device=device),
        velocities=vel,
        density_ratio=torch.full((1,), float(density_ratio), dtype=dtype, device=device),
        fixed=torch.full((1,), bool(fixed), dtype=torch.bool, device=device),
    )


# --------------------------------------------------------------------------- #
# LBM kernels
# --------------------------------------------------------------------------- #

_LATTICE_CACHE: dict = {}


def _lattice(device: torch.device, dtype: torch.dtype):
    key = (str(device), dtype)
    if key not in _LATTICE_CACHE:
        _LATTICE_CACHE[key] = (
            C3D19.to(device=device, dtype=dtype),
            W3D19.to(device=device, dtype=dtype),
        )
    return _LATTICE_CACHE[key]


def _macro_with_half_force(f: torch.Tensor, force: torch.Tensor | None):
    rho, ux, uy, uz = macroscopic3d(f)
    if force is not None:
        inv = 1.0 / torch.clamp(rho, min=1e-12)
        ux = ux + 0.5 * force[0] * inv
        uy = uy + 0.5 * force[1] * inv
        uz = uz + 0.5 * force[2] * inv
    return rho, ux, uy, uz


def _guo_forcing(ux, uy, uz, force, tau, c, w):
    """Second-order Guo forcing (Guo, Zheng & Shi 2002)."""
    q = c.shape[0]
    fx, fy, fz = force[0], force[1], force[2]
    cx = c[:, 0].view(q, 1, 1, 1)
    cy = c[:, 1].view(q, 1, 1, 1)
    cz = c[:, 2].view(q, 1, 1, 1)
    wv = w.view(q, 1, 1, 1)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    cf = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0) + cz * fz.unsqueeze(0)
    uf = (fx * ux + fy * uy + fz * uz).unsqueeze(0)
    return wv * (1.0 - 0.5 / tau) * (3.0 * (cf - cu * uf) + 9.0 * cu * cf)


def lbm_step_bgk_guo(
    f: torch.Tensor, tau: float, force: torch.Tensor | None = None
) -> torch.Tensor:
    """One D3Q19 BGK collide+stream step with optional Guo body force.

    Args:
        f: distribution ``(19, nz, ny, nx)``.
        tau: BGK relaxation time (``nu = (tau - 1/2)/3``).
        force: Eulerian force density ``(3, nz, ny, nx)``.

    Returns:
        Streamed post-collision distribution (periodic boundaries).
    """
    c, w = _lattice(f.device, f.dtype)
    rho, ux, uy, uz = _macro_with_half_force(f, force)
    feq = equilibrium3d(rho, ux, uy, uz)
    f = f + (feq - f) / tau
    if force is not None:
        f = f + _guo_forcing(ux, uy, uz, force, tau, c, w)
    return stream3d(f)


# --------------------------------------------------------------------------- #
# Simulation driver
# --------------------------------------------------------------------------- #


class FCMSimulation:
    """FCM-LBM driver: D3Q19 BGK fluid, Gaussian-coupled spherical particles."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        tau: float,
        *,
        sigma: float = 1.5,
        marker_spacing: float = 1.0,
        rho0: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        u0: float = 0.0,
    ) -> None:
        if tau <= 0.5:
            raise ValueError("tau must exceed 0.5")
        self.shape = tuple(int(n) for n in shape)
        self.tau = float(tau)
        self.nu = (self.tau - 0.5) / 3.0
        self.rho0 = float(rho0)
        self.sigma = float(sigma)
        self.marker_spacing = float(marker_spacing)
        self.dtype = dtype
        self.device = torch.device(device)
        self.u0 = float(u0)
        self.f: torch.Tensor | None = None
        self.coupler: FCMCoupler | None = None
        self.time = 0
        self.last_forces: torch.Tensor | None = None
        self.particles: ParticleSet | None = None
        self.marker_offsets: list[torch.Tensor] = []
        self.marker_pid: torch.Tensor | None = None

    # -- setup ------------------------------------------------------------- #
    def initialize(self, particles: ParticleSet | None = None, u0: float | None = None) -> None:
        us = self.u0 if u0 is None else u0
        nz, ny, nx = self.shape
        rho = torch.full((nz, ny, nx), self.rho0, dtype=self.dtype, device=self.device)
        zero = torch.zeros_like(rho)
        ux = torch.full_like(rho, us)
        self.f = equilibrium3d(rho, ux, zero.clone(), zero.clone())
        self.time = 0
        self.last_forces = None
        self.configure(particles)

    def configure(self, particles: ParticleSet | None) -> None:
        self.particles = particles
        self.marker_offsets = []
        if particles is None or particles.n == 0:
            self.coupler = None
            self.marker_pid = None
            return
        self.coupler = FCMCoupler(
            self.shape, self.sigma, dtype=self.dtype, device=self.device
        )
        pids = []
        for pid in range(particles.n):
            off = sphere_markers(float(particles.radii[pid]), self.marker_spacing)
            self.marker_offsets.append(off.to(self.device))
            pids.append(torch.full((off.shape[0],), pid, dtype=torch.long, device=self.device))
        self.marker_pid = torch.cat(pids) if pids else None

    def marker_positions(self, particles: ParticleSet | None = None) -> torch.Tensor:
        particles = particles if particles is not None else self.particles
        if particles is None or particles.n == 0:
            return torch.zeros(0, 3, dtype=self.dtype, device=self.device)
        nz, ny, nx = self.shape
        if self.marker_pid is not None and self.marker_pid.numel() != sum(
            o.shape[0] for o in self.marker_offsets
        ):
            self.configure(particles)
        offs = torch.cat(self.marker_offsets, dim=0)
        pos = particles.positions[self.marker_pid] + offs
        return torch.stack(
            [pos[:, 0] % nx, pos[:, 1] % ny, pos[:, 2] % nz], dim=1
        )

    # -- diagnostics ------------------------------------------------------- #
    def velocity(self, force: torch.Tensor | None = None):
        return _macro_with_half_force(self.f, force)

    def mean_velocity(self) -> torch.Tensor:
        rho, ux, uy, uz = macroscopic3d(self.f)
        tot = rho.sum()
        return torch.stack([(rho * ux).sum(), (rho * uy).sum(), (rho * uz).sum()]) / tot

    def kinetic_energy(self) -> float:
        rho, ux, uy, uz = macroscopic3d(self.f)
        return float(0.5 * (rho * (ux * ux + uy * uy + uz * uz)).sum())

    def momentum(self) -> torch.Tensor:
        rho, ux, uy, uz = macroscopic3d(self.f)
        return torch.stack([(rho * ux).sum(), (rho * uy).sum(), (rho * uz).sum()])

    def remove_mean_momentum(self) -> torch.Tensor:
        """Subtract the volume-averaged fluid velocity (small-Mach Galilean shift).

        In a closed periodic box with an external force on the particles the
        fluid momentum would otherwise grow without bound; the physical
        equivalent is the frame in which the box-average fluid velocity is zero.
        Applied through the linearised equilibrium correction
        ``f_i <- f_i - 3 w_i rho (c_i . u_mean)`` (``|u_mean| < 1e-3`` in all
        campaign runs, so the O(u^2) error is negligible).
        """
        um = self.mean_velocity()
        c, w = _lattice(self.device, self.dtype)
        q = c.shape[0]
        rho = self.f.sum(dim=0)
        corr = 3.0 * w.view(q, 1, 1, 1) * rho.unsqueeze(0) * (
            c[:, 0].view(q, 1, 1, 1) * um[0]
            + c[:, 1].view(q, 1, 1, 1) * um[1]
            + c[:, 2].view(q, 1, 1, 1) * um[2]
        )
        self.f = self.f - corr
        return um

    # -- one step ---------------------------------------------------------- #
    def step(
        self,
        particles: ParticleSet | None = None,
        *,
        body_force: torch.Tensor | None = None,
        external_force: torch.Tensor | None = None,
        update_particles: bool = True,
        remove_mean_momentum: bool = False,
    ) -> torch.Tensor:
        """Advance one lattice time step and return ``(N, 3)`` particle forces.

        The returned values are the total forces that the particles exert on the
        fluid; the hydrodynamic force on the particle is ``-F_p``.
        """
        if particles is None:
            particles = self.particles
        force = body_force
        fp = None
        if particles is not None and particles.n > 0:
            assert self.coupler is not None
            if self.marker_pid is None or self.marker_pid.shape[0] != sum(
                o.shape[0] for o in self.marker_offsets
            ):
                self.configure(particles)
            _, ux, uy, uz = self.velocity(force)
            u_field = torch.stack([ux, uy, uz])
            mpos = self.marker_positions(particles)
            u_m = self.coupler.interpolate(u_field, mpos)
            u_target = particles.velocities[self.marker_pid]
            f_m = self.rho0 * (u_target - u_m)  # dx^3 = dt = 1
            fp = torch.zeros(particles.n, 3, dtype=self.dtype, device=self.device)
            fp.index_add_(0, self.marker_pid, f_m)
            spread = self.coupler.spread(f_m, mpos)
            self._last_marker_forces = f_m
            self._last_marker_positions = mpos
            force = spread if force is None else force + spread
        self.f = lbm_step_bgk_guo(self.f, self.tau, force)
        self.time += 1
        if remove_mean_momentum:
            self.remove_mean_momentum()
        self.last_forces = fp
        if particles is not None and particles.n > 0 and update_particles and fp is not None:
            self._advance_particles(particles, fp, external_force)
        return fp

    def _advance_particles(
        self, particles: ParticleSet, fp: torch.Tensor, external_force: torch.Tensor | None
    ) -> None:
        mass = particles.masses(self.rho0).view(-1, 1)
        fext = 0.0 if external_force is None else external_force
        acc = (fext - fp) / mass
        moving = (~particles.fixed).view(-1, 1)
        particles.velocities = torch.where(
            moving, particles.velocities + acc, particles.velocities
        )
        newpos = torch.where(
            moving, particles.positions + particles.velocities, particles.positions
        )
        nz, ny, nx = self.shape
        particles.positions = torch.stack(
            [newpos[:, 0] % nx, newpos[:, 1] % ny, newpos[:, 2] % nz], dim=1
        )